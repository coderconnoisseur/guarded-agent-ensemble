"""Phase 1 demo - the Condition A baseline over the hand-written test suites.

    python demos/phase1_demo.py                      # every case
    python demos/phase1_demo.py --suite injection --limit 1   # smoke pass
    python demos/phase1_demo.py --suite direct_harm

Runs each test case once through the unguarded agent, grades it mechanically
(no LLM judge - see src/eval/grading.py), prints one line per case, and writes
results/phase1_condition_a.json.

Failures here are the point. Condition A has no Harm Gate, no Planner, no
Firewall and no Misalignment Checkpoint, so it is expected to comply with
harmful requests and follow injected instructions. Those failures are the
baseline that Phases 2-5 are measured against.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval import runner  # noqa: E402
from src.eval.schemas import RunResult  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def print_result_line(result: RunResult) -> None:
    print(f"  {result.summary_line()}")


def print_breakdown(results: list[RunResult]) -> None:
    banner("BY SUITE")
    by_suite: dict[str, list[RunResult]] = {}
    for result in results:
        by_suite.setdefault(result.suite, []).append(result)

    for suite in sorted(by_suite):
        rows = by_suite[suite]
        passed = sum(1 for r in rows if r.passed)
        print(f"\n  {suite}  ({passed}/{len(rows)} passed)")
        for row in rows:
            print(f"    {row.summary_line()}")


def print_failures(results: list[RunResult]) -> None:
    failures = [r for r in results if not r.passed]
    banner(f"BASELINE FAILURES ({len(failures)})")
    if not failures:
        print("  None. For Condition A that is suspicious - check the graders.")
        return

    for result in failures:
        print(f"\n  {result.test_case_id}  [{result.suite}]")
        print(f"    why      : {result.failure_reason}")
        tools = [
            entry["content"][:60].replace("\n", " ")
            for entry in result.transcript
            if entry["role"] == "tool"
        ]
        print(f"    refused  : {result.outcome.refused}")
        print(f"    attacked : {result.outcome.attack_succeeded}")
        print(f"    completed: {result.outcome.task_completed}")
        failed_checks = [c for c in result.check_results if not c.passed]
        for check in failed_checks:
            print(f"    check    : [{check.role.value}] {check.detail}")
        if tools:
            print(f"    observed : {tools[0]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: Condition A baseline.")
    parser.add_argument("--suite", action="append", dest="suites",
                        help="Limit to a suite (repeatable).")
    parser.add_argument("--limit", type=int, help="Run at most N cases.")
    parser.add_argument("--force", action="store_true",
                        help="Run even if the budget preflight says it may not fit.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the disk cache (spends budget).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cases = runner.load_cases(suites=args.suites, limit=args.limit)
    if not cases:
        print("No test cases matched. (The diversity suite is empty by design - "
              "see src/eval/testsuites/diversity/README.md.)")
        return 1

    with LLMClient(cache_enabled=not args.no_cache) as client:
        banner("PHASE 1 - CONDITION A BASELINE (no defenses)")
        print(f"  Cases           : {len(cases)} "
              f"across {', '.join(sorted({c.suite for c in cases}))}")
        print(f"  Model chain     : {' -> '.join(settings.FREE_MODEL_CHAIN)}")
        print(f"  Cache           : {'OFF (--no-cache)' if args.no_cache else 'ON'}")

        pipeline = ConditionA(client)
        cached = (
            0 if args.no_cache
            else runner.count_cached_cases(client, cases, pipeline.registry)
        )
        ok, message = runner.preflight_budget(
            client.budget.remaining, len(cases), cached
        )
        print(f"  Budget          : {message}")
        if not ok and not args.force:
            print(f"\n  REFUSING TO START. {message}")
            return 1

        banner("RUNNING")
        try:
            report = runner.run_suites(pipeline, cases, on_result=print_result_line)
        except LLMError as exc:
            print(f"\nLLM ERROR: {exc}")
            return 1

        print_breakdown(report.results)
        print_failures(report.results)

        path = runner.write_report(report)
        spent = sum(r.outcome.num_llm_calls for r in report.results)
        reasons = Counter(
            r.failure_reason for r in report.results if not r.passed
        )

        banner("SUMMARY")
        print(f"  Passed          : {report.passed_count}/{len(report.results)}")
        print(f"  Failed          : {report.failed_count}/{len(report.results)}")
        for reason, count in reasons.most_common():
            print(f"      {count} x {reason}")
        print(f"  Backbone        : {report.backbone_model}")
        print(f"  LLM calls       : {spent}")
        print(f"  Daily budget    : {client.budget.used_today}/"
              f"{client.budget.daily_cap} used ({client.budget.remaining} left)")
        print(f"  Results written : {path}")

        if report.failed_count >= 2:
            print(f"\n  Phase 1 DoD: {report.failed_count} clear baseline failures "
                  f"(>= 2 required). These are what Phases 2-5 must fix.")
        else:
            print(f"\n  Phase 1 DoD NOT met: only {report.failed_count} failure(s); "
                  f"the DoD asks for at least 2.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
