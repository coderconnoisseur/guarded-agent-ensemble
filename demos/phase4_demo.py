"""Phase 4 demo - Response Firewall + Quarantine, side by side.

    python demos/phase4_demo.py                 # inj_008, the flagship case
    python demos/phase4_demo.py --case inj_001
    python demos/phase4_demo.py --ablation

CLAUDE.md 10 calls this the single most important demo in the project. It runs
one injection case through both conditions and prints the transcripts side by
side: Condition A gets hijacked, Condition B flags the response, quarantines
it, replays a sanitised substitute, and still finishes the user's real task.

The default runs inj_005 with **only** the Firewall and Quarantine enabled -
the Planner is deliberately switched off. That is 9.1 tier 2's single-module
isolation, and it is the only honest way to show what this module contributes
by itself: inj_005 is the one injection case the unguarded backbone actually
follows, so it is the case where "did the defense help" has an answer.

Leaving the Planner on would make the demo unfalsifiable - the Planner already
blocks that call, so Condition B would pass whether or not the Firewall works.

`--case inj_008` shows the complementary situation: an injection that reuses a
tool the user's own task legitimately needs, where the Planner cannot help
because comms.send_email is in the plan. The Firewall flags and quarantines it
correctly - though on this backbone the agent resists that payload unaided, so
it demonstrates the mechanism rather than a behavioural improvement.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval import runner  # noqa: E402
from src.eval.schemas import RunResult, TestCase  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

WIDTH = 46
RULE = "=" * (WIDTH * 2 + 3)
DEFAULT_CASE = "inj_005"


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def column(text: str) -> list[str]:
    """Wrap one side of the side-by-side view."""
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, WIDTH) or [""])
    return lines


def side_by_side(left_title: str, left: str, right_title: str, right: str) -> None:
    print(f"  {left_title:<{WIDTH}} | {right_title}")
    print(f"  {'-' * WIDTH}-+-{'-' * WIDTH}")
    left_lines, right_lines = column(left), column(right)
    for i in range(max(len(left_lines), len(right_lines))):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        print(f"  {l:<{WIDTH}} | {r}")


def transcript_text(result: RunResult) -> str:
    """Condense a transcript into the parts a reader needs."""
    out: list[str] = []
    for entry in result.transcript:
        role = entry["role"]
        content = entry["content"].strip()
        if role == "system":
            continue
        if role == "user" and content.startswith("Observation:"):
            role = "tool"
            content = content[len("Observation:"):].strip()
        label = {"user": "USER", "assistant": "AGENT", "tool": "TOOL"}.get(
            role, role.upper()
        )
        out.append(f"[{label}]")
        out.append(content[:700])
        out.append("")
    return "\n".join(out)


def tools_used(result: RunResult) -> list[str]:
    text = " ".join(
        e["content"] for e in result.transcript if e["role"] == "assistant"
    )
    return re.findall(r'"tool"\s*:\s*"([^"]+)"', text)


def print_defense_activity(result: RunResult) -> None:
    banner("WHAT THE DEFENSES DID (Condition B)")
    if not result.plan_ran:
        print("  Planner    : DISABLED for this run, so the Firewall's own "
              "contribution is visible")
    else:
        print(f"  Planner    : {result.plan_node_count} node(s) planned, "
              f"executed {result.plan_executed or '[]'}")
        if result.plan_rejections:
            print(f"               BLOCKED off-plan: {result.plan_rejections}")
        else:
            print("               nothing blocked - the tool was legitimately "
                  "planned, so only response scanning could catch this")
    print(f"  Firewall   : {'FLAGGED' if result.firewall_flagged else 'clean'}"
          f"  stages={result.firewall_stages or '[]'}")
    if result.firewall_signals:
        print(f"               signals: {', '.join(result.firewall_signals)}")
    print(f"  Quarantine : {result.quarantined_tools or '[]'}"
          f"  ({result.quarantine_removed_lines} line(s) removed)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: Firewall + Quarantine.")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument(
        "--modules", default="firewall,quarantine",
        help="Modules to enable in Condition B (default: firewall,quarantine - "
             "the Planner is off so the Firewall's own contribution is visible).",
    )
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cases = {c.id: c for c in runner.load_cases()}
    case: TestCase | None = cases.get(args.case)
    if case is None:
        print(f"No case {args.case!r}. Available: {', '.join(sorted(cases))}")
        return 1

    with LLMClient() as client:
        banner("PHASE 4 - RESPONSE FIREWALL + QUARANTINE (ShieldMCP + IPIGuard)")
        print(f"  Backbone   : {args.model}  (pinned)")
        print(f"  Case       : {case.id}  [{case.category}]")
        print(f"  User asked : {case.prompt}")
        print(f"  Payload in : {case.injection_tool} response "
              f"(planted before the run; the user never said it)")
        print(f"  B modules  : {args.modules}")

        try:
            result_a = runner.run_case(
                ConditionA(client, model=args.model), case
            )
            result_b = runner.run_case(
                ConditionB(client, model=args.model,
                           enabled_modules=set(args.modules.split(","))),
                case,
            )
        except LLMError as exc:
            print(f"\nLLM ERROR: {exc}")
            return 1

        banner("THE INJECTED PAYLOAD (what the tool returned)")
        for line in textwrap.wrap(case.injected_content or "", WIDTH * 2):
            print(f"  {line}")

        banner("TRANSCRIPTS, SIDE BY SIDE")
        side_by_side(
            "CONDITION A - unguarded", transcript_text(result_a),
            "CONDITION B - guarded", transcript_text(result_b),
        )

        print_defense_activity(result_b)

        banner("OUTCOME")
        print(f"  {'':12} {'Condition A':>28} {'Condition B':>28}")
        rows = [
            ("tools used", ", ".join(tools_used(result_a)) or "-",
             ", ".join(tools_used(result_b)) or "-"),
            ("attack succeeded", str(result_a.outcome.attack_succeeded),
             str(result_b.outcome.attack_succeeded)),
            ("task completed", str(result_a.outcome.task_completed),
             str(result_b.outcome.task_completed)),
            ("verdict", "PASS" if result_a.passed else "FAIL",
             "PASS" if result_b.passed else "FAIL"),
        ]
        for label, a, b in rows:
            print(f"  {label:16} {a[:28]:>28} {b[:28]:>28}")
        if not result_a.passed:
            print(f"\n  A failed because : {result_a.failure_reason}")
        if not result_b.passed:
            print(f"  B failed because : {result_b.failure_reason}")

        banner("WHAT THIS SHOWS")
        if result_b.firewall_flagged and result_b.outcome.task_completed:
            print("  The firewall judged the tool response to be instructions rather")
            print("  than information, quarantine replayed it with the injected lines")
            print("  removed, and the agent finished the user's real task from the")
            print("  sanitised content. Blocking alone would have denied the user")
            print("  their task - which is the attacker winning by other means.")
        elif result_b.firewall_flagged:
            print("  The firewall flagged the response, but the task did not complete.")
            print("  Quarantine is supposed to preserve the legitimate content - check")
            print("  how much sanitise() removed.")
        else:
            print("  The firewall did not flag this response. Either the payload is")
            print("  subtler than the heuristic and the guard model handle, or the")
            print("  case does not carry one.")

        print(f"\n  Daily budget : {client.budget_summary()}")

        if args.ablation:
            banner("ABLATION SNAPSHOT (9.1 tier 1) - full mixed suite")
            report = runner.run_suites(
                ConditionB(client, model=args.model,
                           enabled_modules={"harm_gate", "planner",
                                            "firewall", "quarantine"}),
                runner.load_cases(),
                on_result=lambda r: print(f"  {r.summary_line()}"),
            )
            report.notes.append(
                "modules enabled: harm_gate, planner, firewall, quarantine"
            )
            out = settings.RESULTS_DIR / "ablation_phase4.json"
            report.write(out)
            print(f"\n  Wrote {out.name} "
                  f"({report.passed_count}/{len(report.results)} passed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
