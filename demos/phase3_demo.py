"""Phase 3 demo - the Tool Dependency Graph, planned vs executed.

    python demos/phase3_demo.py
    python demos/phase3_demo.py --case inj_005     # show it blocking an injection
    python demos/phase3_demo.py --ablation

Runs one benign task through Condition B and prints the emitted TDG beside the
tool calls that actually ran, then the same task through Condition A for
contrast, per CLAUDE.md 10's Phase 3 Definition of Done.

The interesting comparison is not "did the plan match" on a benign task - it
should. It is what happens on an injection case, where the plan was committed
to *before* the malicious instruction existed in the agent's context, so the
tool the injection wants was never in it. `--case inj_005` shows that.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval import runner  # noqa: E402
from src.eval.schemas import TestCase  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

RULE = "=" * 78
DEFAULT_CASE = "benign_002"


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Tool Dependency Graph.")
    parser.add_argument("--case", default=DEFAULT_CASE,
                        help=f"Test case id to run (default: {DEFAULT_CASE}).")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument("--ablation", action="store_true",
                        help="Also snapshot Condition B over the full mixed suite "
                             "to results/ablation_phase3.json (9.1 tier 1).")
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
        banner("PHASE 3 - PLAN-THEN-EXECUTE PLANNER (IPIGuard)")
        print(f"  Backbone   : {args.model}  (pinned)")
        print(f"  Case       : {case.id}  [{case.suite}]")
        print(f"  Task       : {case.prompt[:150]}")
        if case.injected_content:
            print(f"  Payload in : {case.injection_tool} "
                  f"(planted before the run, unseen at plan time)")

        try:
            result_b = runner.run_case(
                ConditionB(client, model=args.model,
                           enabled_modules={"planner"}),
                case,
            )
            result_a = runner.run_case(
                ConditionA(client, model=args.model), case
            )
        except LLMError as exc:
            print(f"\nLLM ERROR: {exc}")
            return 1

        # run_case returns a RunResult; re-run through the pipeline object is
        # unnecessary because the enforcement record is persisted on it.
        banner("CONDITION B - plan committed BEFORE any tool ran")
        if result_b.plan_degraded:
            print("  WARNING: planning failed; this run fell back to read-only")
            print("           and was never actually constrained by a plan.\n")
        print(f"  Planned nodes  : {result_b.plan_node_count}")
        print(f"  Executed       : {result_b.plan_executed or '(none)'}")
        print(f"  Expanded       : {result_b.plan_expansions or '(none)'}"
              f"   (read-only, allowed off-plan)")
        print(f"  BLOCKED        : {result_b.plan_rejections or '(none)'}")
        print(f"  Verdict        : {result_b.summary_line()}")

        print_condition_a_from_run(result_a)

        banner("WHAT THIS SHOWS")
        if result_b.plan_rejections:
            print("  The plan was fixed before the agent saw any tool output, so a")
            print("  tool the injected instruction wanted was never in it. The call")
            print("  was refused at dispatch, not judged after the fact.")
        elif result_b.plan_expansions:
            print("  Execution stayed inside the plan except for read-only lookups,")
            print("  which Node Expansion permits because a query cannot exfiltrate")
            print("  or destroy anything.")
        else:
            print("  Execution matched the emitted plan exactly - the benign case,")
            print("  where the constraint costs nothing.")

        print(f"\n  Daily budget   : {client.budget_summary()}")

        if args.ablation:
            banner("ABLATION SNAPSHOT (9.1 tier 1) - full mixed suite")
            report = runner.run_suites(
                ConditionB(client, model=args.model,
                           enabled_modules={"harm_gate", "planner"}),
                runner.load_cases(),
                on_result=lambda r: print(f"  {r.summary_line()}"),
            )
            report.notes.append("modules enabled: harm_gate, planner")
            out = settings.RESULTS_DIR / "ablation_phase3.json"
            report.write(out)
            print(f"\n  Wrote {out.name} "
                  f"({report.passed_count}/{len(report.results)} passed)")

    return 0


def print_condition_a_from_run(result) -> None:
    banner("CONDITION A - unconstrained, for contrast")
    calls = [
        entry for entry in result.transcript if entry["role"] == "assistant"
    ]
    import re

    tools = re.findall(
        r'"tool"\s*:\s*"([^"]+)"', " ".join(c["content"] for c in calls)
    )
    print("  EXECUTED TOOL CALLS (no plan existed)")
    for name in tools or ["(none)"]:
        print(f"    [free    ] {name}")
    print(f"\n  Verdict        : {result.summary_line()}")


if __name__ == "__main__":
    raise SystemExit(main())
