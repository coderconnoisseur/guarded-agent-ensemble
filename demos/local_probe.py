"""Can a local model actually be this project's backbone? (CLAUDE.md §11)

    python demos/local_probe.py                    # probe the configured chain
    python demos/local_probe.py --model qwen3:4b   # probe one model
    python demos/local_probe.py --dry-run          # what it would do, no calls

WHY THIS EXISTS RATHER THAN JUST SWAPPING THE MODEL
---------------------------------------------------
A smaller backbone is wanted here *because* it is weaker: the pinned Groq
model resists AgentDojo's own five attack templates 0/30 and sits at
`ASR_inj` 0.07, which needs n=65 to reach significance (`stats.required_n`)
against the 15 injection cases that exist. At a baseline of 0.40 it needs
n=9. Failure is the scarce resource.

But there is a floor, and it is measured rather than guessed.
`liquid/lfm-2.5-2.6b` was dropped from `FREE_MODEL_CHAIN` because at 2.6B it
could not reliably emit the Tool Dependency Graph Phase 3 needs. A backbone
below that floor does not produce a weak Condition B - it produces **no**
Condition B, because the Planner degrades, the ReAct parser rejects the
replies, and the ablation collapses into noise that looks like a result.

So this probe answers the only question worth asking before spending hours on
a grid: **can this model speak the protocol at all?** Three checks, cheapest
first, each one a hard gate:

  1. **Reachable** - is a server there, and does it have the model?
  2. **ReAct protocol** (CLAUDE.md §5.3) - does it emit a parseable
     `Thought:/Action:` or `Final:` reply? If not, nothing else matters.
  3. **Tool Dependency Graph** (CLAUDE.md §3, IPIGuard) - does it emit a
     valid JSON plan over the real tool registry, without the Planner having
     to degrade? This is the check the 2.6B model failed.

A model that passes all three is worth running the suite against. One that
fails any of them is reported as failing *that* check, not as "bad" - the
distinction is what tells you whether to drop a size or drop the idea.

This probe deliberately does **not** score `HS`, `ASR_inj` or any GAI term.
Those are the grid's job, and measuring them here on a handful of cases is
exactly the tune-and-report-on-the-same-data trap the Harm Gate already fell
into once (HANDOFF §5.2d).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import httpx  # noqa: E402

from config import settings  # noqa: E402
from src.agent.loop import parse_agent_reply  # noqa: E402
from src.agent.prompts import build_system_prompt  # noqa: E402
from src.defense.planner import Planner  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.tools.registry import build_registry  # noqa: E402

RULE = "=" * 84

# A task with an unambiguous two-step shape: read something, then act on it.
# Chosen so a correct plan is obvious to a human reading the output, rather
# than something only the validator can judge.
PROBE_TASK = "Read the welcome file in the sandbox and email a summary to alex@example.com"


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def check_reachable(model: str) -> tuple[bool, str]:
    """Is an Ollama server up, and does it have this model pulled?

    Separated from the protocol checks because the failure modes need
    completely different responses: "start Ollama" and "pull the model" are
    not "this model cannot be the backbone".
    """
    root = settings.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
    try:
        reply = httpx.get(f"{root}/api/tags", timeout=5.0)
        reply.raise_for_status()
    except httpx.HTTPError as exc:
        return False, (
            f"no Ollama server at {root} ({type(exc).__name__}). "
            f"Start it, or set OLLAMA_BASE_URL."
        )
    names = [m.get("name", "") for m in reply.json().get("models", [])]
    # Ollama reports "qwen3:4b" but also answers to it without the tag, so
    # match on the prefix rather than demanding an exact string.
    if not any(n == model or n.startswith(model.split(":")[0] + ":") for n in names):
        return False, (
            f"server is up but has no {model!r}. Pulled models: "
            f"{', '.join(names) or '(none)'}. Run: ollama pull {model}"
        )
    exact = [n for n in names if n == model]
    if not exact:
        return True, (
            f"server is up; {model!r} not pulled exactly but a same-family "
            f"tag exists ({', '.join(names)}). Ollama may still resolve it."
        )
    return True, f"server is up and {model} is present"


def check_react_protocol(client: LLMClient, model: str) -> tuple[bool, str, str]:
    """Does the model emit a reply the agent loop can parse (CLAUDE.md §5.3)?

    Uses the project's real parser rather than a lookalike regex. A probe that
    accepts something `parse_agent_reply` would reject is worse than no probe.
    """
    registry = build_registry("workspace")
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(registry.describe_for_prompt()),
        },
        {"role": "user", "content": PROBE_TASK},
    ]
    try:
        response = client.chat(messages, model=model)
    except LLMError as exc:
        return False, f"the call itself failed: {exc}", ""

    parsed = parse_agent_reply(response.content)
    excerpt = response.content.strip()[:300]
    if parsed.kind == "unparseable":
        return False, f"reply did not parse: {parsed.problem}", excerpt
    if parsed.kind == "action":
        return True, f"emitted a parseable Action: {parsed.tool}", excerpt
    return True, "emitted a parseable Final (no tool call attempted)", excerpt


def check_tdg(client: LLMClient, model: str) -> tuple[bool, str, str]:
    """Does the model emit a usable Tool Dependency Graph (CLAUDE.md §3)?

    The check the 2.6B model failed. `build_plan` never raises - it degrades
    to a read-only plan so a planning failure cannot also destroy the agent -
    so `degraded` is the signal, not an exception. An empty or degraded graph
    means the Planner is not constraining anything, and Condition B would be
    Condition A wearing a costume.
    """
    registry = build_registry("workspace")
    planner = Planner(client, model=model)
    try:
        graph = planner.build_plan(PROBE_TASK, registry)
    except LLMError as exc:
        return False, f"the call itself failed: {exc}", ""

    rendered = json.dumps(
        [{"tool": n.tool, "depends_on": list(n.depends_on)} for n in graph.nodes],
        indent=2,
    )
    if graph.degraded:
        return False, (
            "the planner DEGRADED - the model could not produce a valid plan, "
            "so Condition B would not actually be constrained"
        ), rendered
    if not graph.nodes:
        return False, "the plan was empty; nothing would be permitted", rendered
    return True, f"emitted a valid {len(graph.nodes)}-node plan", rendered


def probe(client: LLMClient, model: str) -> bool:
    banner(f"PROBING {model}")

    ok, detail = check_reachable(model)
    print(f"  [{'PASS' if ok else 'FAIL'}] reachable        - {detail}")
    if not ok:
        return False

    started = time.monotonic()
    ok_react, detail, excerpt = check_react_protocol(client, model)
    react_s = time.monotonic() - started
    print(f"  [{'PASS' if ok_react else 'FAIL'}] ReAct protocol   - {detail} "
          f"({react_s:.1f}s)")
    if excerpt:
        print("        reply: " + excerpt.replace("\n", "\n               "))

    started = time.monotonic()
    ok_tdg, detail, rendered = check_tdg(client, model)
    tdg_s = time.monotonic() - started
    print(f"  [{'PASS' if ok_tdg else 'FAIL'}] tool dep. graph  - {detail} "
          f"({tdg_s:.1f}s)")
    if rendered:
        print("        plan:  " + rendered.replace("\n", "\n               "))

    passed = ok_react and ok_tdg
    print()
    if passed:
        print(f"  {model} can speak the protocol. It is a usable backbone arm.")
        print(f"  Roughly {react_s:.0f}s per agent turn and {tdg_s:.0f}s per plan; "
              f"at ~11.6 calls/case that is ~{(react_s * 11.6) / 60:.0f} min "
              f"per Condition B case.")
    elif not ok_tdg and ok_react:
        print(f"  {model} talks but cannot plan. This is the failure that "
              f"retired liquid/lfm-2.5-2.6b:")
        print("  it would give a degraded Condition B, not a weak one. Try the "
              "next model up.")
    else:
        print(f"  {model} cannot speak the protocol at all. Not a candidate.")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a local backbone.")
    parser.add_argument("--model", default="",
                        help="One model to probe; default is the configured chain.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Say what would be probed; make no calls.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    candidates = [args.model] if args.model else list(settings.OLLAMA_MODEL_CHAIN)

    banner("LOCAL BACKBONE PROBE (CLAUDE.md §11)")
    print(f"  Server     : {settings.OLLAMA_BASE_URL}")
    print(f"  Candidates : {', '.join(candidates)}")
    print(f"  Reply cap  : {settings.OLLAMA_MIN_MAX_TOKENS} tokens "
          f"(vs {settings.DEFAULT_MAX_TOKENS} hosted; thinking cannot be "
          f"disabled, so the budget must cover it)")
    print(f"  Pinned arm : {settings.BACKBONE_MODEL} (unchanged; this is a "
          f"second arm)")
    print()
    print("  Three gates, cheapest first: reachable -> ReAct protocol -> TDG.")
    print("  The floor is measured: a 2.6B model was retired for failing the")
    print("  third gate, and a backbone that cannot plan gives no Condition B.")

    if args.dry_run:
        print("\n  --dry-run: stopping before any call.")
        return 0

    # Cache off: a probe is asking what the model does *now*, and a cached
    # reply from a different model would answer a question nobody asked.
    with LLMClient(
        model_chain=[("ollama", m) for m in candidates], cache_enabled=False
    ) as client:
        for model in candidates:
            if probe(client, model):
                banner("RESULT")
                print(f"  Use {model}. To run against it:")
                print(f"    USE_LOCAL_BACKBONE=1 LOCAL_BACKBONE_MODEL={model} \\")
                print("      python demos/frozen_ablation.py --dry-run")
                print()
                print("  Nothing already in results/ is affected: the pinned")
                print(f"  {settings.BACKBONE_MODEL} arm is untouched and this is a")
                print("  second arm, not a replacement.")
                return 0

    banner("RESULT")
    print("  No candidate passed. Options, in order of preference:")
    print("    1. Pull the next size up and re-probe.")
    print("    2. Keep the hosted backbone and accept the 2/min rate limit.")
    print("  Do NOT drop below ~3B to make something fit: that trades a slow")
    print("  measurement for an unmeasurable one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
