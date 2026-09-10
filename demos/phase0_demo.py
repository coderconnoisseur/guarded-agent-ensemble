"""Phase 0 demo - the bare ReAct agent, end to end.

    python demos/phase0_demo.py "read the sandbox welcome file and summarize it"

Prints the full transcript, the tools actually dispatched, and the cache state
of every LLM call. Run it twice: the second run should report `from_cache=True`
for each call and spend nothing from the daily budget, which is the Phase 0
Definition of Done in CLAUDE.md 10.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which mangles the curly quotes and dashes
# models routinely emit ("I’ll" -> "I?ll"). The data is fine on disk; only the
# display breaks. Phase 4's side-by-side transcripts have to be legible, so
# every demo forces UTF-8 out.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.agent.loop import AgentResult, ReActAgent  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.tools.files import ensure_sandbox  # noqa: E402
from src.tools.registry import build_default_registry  # noqa: E402

DEFAULT_TASK = "read the sandbox welcome file and summarize it"
RULE = "=" * 78
THIN = "-" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def print_transcript(result: AgentResult) -> None:
    banner("TRANSCRIPT")
    for entry in result.transcript:
        role = entry["role"].upper()
        content = entry["content"].strip()
        if role == "SYSTEM":
            # The tool catalogue is long and identical every run; a pointer is
            # more useful in the demo output than 40 lines of schema.
            first = content.splitlines()[0] if content.splitlines() else ""
            print(f"\n[{role}] {first}")
            print(f"         ... ({len(content)} chars of protocol + tool catalogue)")
            continue
        print(f"\n[{role}]")
        for line in content.splitlines():
            print(f"  {line}")


def print_steps(result: AgentResult) -> None:
    banner("STEP-BY-STEP")
    if not result.steps:
        print("  (no steps taken)")
        return
    for step in result.steps:
        cache = "CACHED" if step.from_cache else f"{step.latency_ms}ms"
        print(f"\n  Step {step.index}  [{cache}]  model={step.model_used}")
        if step.parsed.thought:
            print(f"    Thought : {step.parsed.thought}")
        if step.parsed.kind == "action":
            flag = "  (CRITICAL)" if step.tool_is_critical else ""
            print(f"    Action  : {step.parsed.tool}{flag}")
            print(f"    Args    : {step.parsed.args}")
            if step.tool_result is not None:
                status = "ok" if step.tool_result.ok else f"ERROR {step.tool_result.error}"
                preview = step.tool_result.content.replace("\n", " ")[:90]
                print(f"    Result  : {status} ({step.tool_result.latency_ms}ms)")
                if step.tool_result.ok:
                    print(f"    Preview : {preview}...")
        elif step.parsed.kind == "final":
            print("    Final   : (task complete)")
        else:
            print(f"    PARSE FAILURE: {step.parsed.problem}")


def print_summary(result: AgentResult, client: LLMClient) -> None:
    banner("SUMMARY")
    fresh = result.num_llm_calls - result.num_cache_hits
    print(f"  Task            : {result.task}")
    print(f"  Stop reason     : {result.stop_reason}")
    print(f"  LLM calls       : {result.num_llm_calls}  "
          f"(cache hits: {result.num_cache_hits}, billed: {fresh})")
    print(f"  Backbone        : {', '.join(result.models_used) or '(none)'}")
    if len(result.models_used) > 1:
        print("  NOTE            : more than one model used - the fallback chain fired.")
    print(f"  Tools called    : "
          f"{', '.join(name for name, _ in result.tool_calls) or '(none)'}")
    print(f"  Wall time       : {result.total_latency_ms} ms")
    # Per-backend, because `client.budget` alone reports whichever provider
    # the chain pointer happens to sit on - which is misleading when a run
    # pinned a different one.
    print(f"  Daily budget    : {client.budget_summary()}")
    if result.error:
        print(f"  Error           : {result.error}")

    banner("FINAL ANSWER")
    print(result.final_answer or "(none)")

    if result.num_cache_hits == result.num_llm_calls and result.num_llm_calls:
        print("\n  [cache] Every LLM call was served from disk - "
              "this run cost 0 requests.")
    elif result.num_cache_hits:
        print(f"\n  [cache] {result.num_cache_hits} of {result.num_llm_calls} "
              f"calls served from disk.")
    else:
        print("\n  [cache] All calls were fresh. Re-run this command to see "
              "them served from cache.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: bare ReAct agent demo.")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Task for the agent.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the disk cache (spends budget)."
    )
    parser.add_argument(
        "--model",
        help="Pin one backbone instead of walking the provider chain "
             "(e.g. gemini-2.5-flash). Scored runs should always pin.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    # The agent's own INFO lines are the interesting ones; httpx narrates every
    # request at INFO too, which drowns them out.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sandbox = ensure_sandbox()
    registry = build_default_registry()

    banner("PHASE 0 - BARE REACT AGENT (Condition A, no defenses)")
    print(f"  Sandbox         : {sandbox}")
    print(f"  Tools registered: {len(registry.names())} -> {', '.join(registry.names())}")
    print(f"  Critical tools  : {', '.join(registry.critical_tools())}")
    chain = (f"PINNED {args.model}" if args.model
             else " -> ".join(f"{p}/{m}" for p, m in settings.PROVIDER_CHAIN))
    print(f"  Model chain     : {chain}")
    print(f"  Cache           : {'OFF (--no-cache)' if args.no_cache else 'ON'}")
    print(f"  Task            : {args.task}")

    with LLMClient(cache_enabled=not args.no_cache) as client:
        agent = ReActAgent(client, registry, **({"model": args.model} if args.model else {}))
        try:
            result = agent.run(args.task)
        except LLMError as exc:
            print(f"\nLLM ERROR: {exc}")
            return 1

        print_transcript(result)
        print_steps(result)
        print_summary(result, client)

    return 0 if result.stop_reason == "final" else 2


if __name__ == "__main__":
    raise SystemExit(main())
