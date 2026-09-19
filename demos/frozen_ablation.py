"""The frozen-suite cumulative ablation (CLAUDE.md 9.1 tier 1, HANDOFF 5.1).

    python demos/frozen_ablation.py            # all five rows, resuming
    python demos/frozen_ablation.py --dry-run  # cost estimate only
    python demos/frozen_ablation.py --force    # ignore rows already written

This is the command HANDOFF 5.1 has been asking for. Every previous ablation
row was taken in a different phase, over whatever the suite happened to
contain that week, so `ablation_table.py` correctly refused to draw a trend
through them: a difference between two rows measured over different cases is
partly a change of test, not a change of defense.

Here all five configurations run over **one frozen case set, in one sitting,
on one pinned backbone**. The case list is resolved once up front and handed
to every row, and each row records the exact ids it covered so the claim is
checkable from the saved files rather than trusted.

WHY IT RESUMES BY DEFAULT
-------------------------
Measured: ~507 fresh LLM calls, ~4.2 hours of wall clock. Not because of the
1000/day request cap - that is not the binding constraint - but because Groq
charges the *requested* max_tokens against a 1000 output-tokens-per-minute
ceiling, which works out at 2 requests/minute (see settings.py). A job that
long will be interrupted, so a row already on disk is skipped unless --force.
Rows are independent, so resuming costs nothing beyond what is left.
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
from src.llm.client import BudgetExceededError, LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

RULE = "=" * 84

# Cumulative configurations, in the order the phases added them. The labels
# match ablation_table.py's SNAPSHOTS so the two agree on what a row is.
ROWS: list[tuple[str, str, set[str] | None]] = [
    ("Condition A (no defenses)", "frozen_1_condition_a", None),
    ("+ Harm Gate", "frozen_2_harm_gate", {"harm_gate"}),
    ("+ Harm Gate + Planner", "frozen_3_planner", {"harm_gate", "planner"}),
    ("+ Harm Gate + Planner + Firewall/Quarantine", "frozen_4_firewall",
     {"harm_gate", "planner", "firewall", "quarantine"}),
    ("+ everything (Condition B)", "frozen_5_everything",
     {"harm_gate", "planner", "firewall", "quarantine", "misalignment"}),
]

# Measured per configuration on 2026-09-11 (HANDOFF 5.2a). Used only for the
# preflight estimate.
MEASURED_CALLS_PER_CASE = [2.8, 2.3, 7.1, 8.9, 11.6]


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def row_path(stem: str) -> Path:
    return settings.RESULTS_DIR / f"{stem}.json"


def estimate(cases: list[TestCase], client: LLMClient) -> None:
    """Print what this will cost before spending any of it."""
    from src.tools.registry import build_registry

    by_scenario: dict[str, list[TestCase]] = {}
    for case in cases:
        by_scenario.setdefault(case.scenario, []).append(case)
    cached = sum(
        runner.count_cached_cases(client, group, build_registry(scenario))
        for scenario, group in by_scenario.items()
    )

    rate = settings.GROQ_RATE_LIMIT_PER_MINUTE
    total = 0.0
    print(f"  {len(cases)} frozen cases, {cached} starting from cache")
    print(f"\n  {'row':46} {'est. calls':>11} {'est. time':>10}")
    for (label, stem, _), per_case in zip(ROWS, MEASURED_CALLS_PER_CASE):
        if row_path(stem).exists():
            print(f"  {label:46} {'(done)':>11} {'-':>10}")
            continue
        calls = len(cases) * per_case
        total += calls
        print(f"  {label:46} {calls:>11.0f} {calls / rate / 60:>9.1f}h")
    print(f"  {'TOTAL (upper bound, ignores cache)':46} "
          f"{total:>11.0f} {total / rate / 60:>9.1f}h")
    print(f"\n  Bound by {rate} requests/minute, not by the "
          f"{settings.GROQ_DAILY_REQUEST_CAP}/day cap.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-suite cumulative ablation.")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cost estimate and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run rows already written.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Resolved ONCE. Every row gets this identical list - that is the whole
    # point of the exercise, and re-loading per row would let an edit between
    # rows reintroduce exactly the incomparability this fixes.
    cases = runner.load_cases()
    frozen_ids = sorted(c.id for c in cases)

    with LLMClient() as client:
        banner("FROZEN-SUITE CUMULATIVE ABLATION (9.1 tier 1)")
        print(f"  Backbone   : {args.model}  (pinned)")
        print(f"  Case set   : {len(cases)} cases, frozen for all five rows")
        print(f"  Scenarios  : {', '.join(sorted({c.scenario for c in cases}))}")
        print(f"  Suites     : {', '.join(sorted({c.suite for c in cases}))}")

        banner("COST")
        estimate(cases, client)
        if args.dry_run:
            return 0

        for index, (label, stem, modules) in enumerate(ROWS, start=1):
            path = row_path(stem)
            if path.exists() and not args.force:
                print(f"\n  [{index}/5] {label} - already on disk, skipping "
                      f"({path.name})", flush=True)
                continue

            banner(f"[{index}/5] {label}")
            pipeline = (
                ConditionA(client, model=args.model) if modules is None
                else ConditionB(client, model=args.model, enabled_modules=modules)
            )
            try:
                report = runner.run_suites(
                    pipeline, cases,
                    on_result=lambda r: print(f"  {r.summary_line()}", flush=True),
                )
            except BudgetExceededError as exc:
                print(f"\n  BUDGET EXHAUSTED: {exc}")
                print(f"  Rows written so far are on disk. Re-run tomorrow to "
                      f"continue from row {index}.")
                return 1
            except LLMError as exc:
                print(f"\n  LLM ERROR on row {index}: {exc}")
                print("  Earlier rows are on disk; re-run to continue.")
                return 1

            report.notes.append(
                "modules enabled: "
                + (", ".join(sorted(modules)) if modules else "(none - Condition A)")
            )
            report.notes.append(f"frozen case set ({len(frozen_ids)}): "
                                + ", ".join(frozen_ids))
            report.write(path)
            print(f"\n  Wrote {path.name} "
                  f"({report.passed_count}/{len(report.results)} passed)")
            print(f"  Budget: {client.budget_summary()}", flush=True)

        banner("DONE")
        print("  All five rows are over the same frozen case set, so")
        print("  `python demos/ablation_table.py` should now accept the trend.")
        print(f"  Budget: {client.budget_summary()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
