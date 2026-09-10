"""Phase 2 demo - Harm Gate: HS_A vs HS_B on the direct-harm suite.

    python demos/phase2_demo.py
    python demos/phase2_demo.py --rubric-only      # strict architecture.md reading
    python demos/phase2_demo.py --ablation         # + full mixed suite snapshot

Runs the direct-harm cases through Condition A (unguarded) and Condition B
(Harm Gate only) on the same pinned backbone, and prints the Harm Score for
each, per CLAUDE.md 10's Phase 2 Definition of Done.

HS is never shown alone. An agent that refuses everything scores HS = 0.00 and
is useless, so BU and the over-refusal rate are printed beside it - that pair
is what 9's weighting exists to protect.
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
from src.eval.scorer import ConditionScore, score_condition  # noqa: E402
from src.eval.schemas import RunResult  # noqa: E402
from src.llm.client import LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

RULE = "=" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def print_gate_activity(results: list[RunResult]) -> None:
    """Which stage of the gate did the work, per case.

    Worth showing explicitly: if every block came from the rubric, the
    classifier is decoration; if every block came from the classifier, the
    rubric is. The split is the evidence for the two-stage design.
    """
    banner("HARM GATE ACTIVITY (Condition B)")
    print(f"  {'case':12} {'expected':9} {'outcome':10} {'blocked at':12} category")
    for row in results:
        expected = "refuse" if row.expects_refusal else "comply"
        outcome = "refused" if row.outcome.refused else "complied"
        stage = row.harm_gate_stage or "-"
        category = row.harm_gate_category or "-"
        mark = " " if (row.outcome.refused == row.expects_refusal) else "!"
        print(f" {mark}{row.test_case_id:12} {expected:9} {outcome:10} "
              f"{stage:12} {category}")


def print_scores(a: ConditionScore, b: ConditionScore) -> None:
    banner("PHASE 2 DEFINITION OF DONE - HS_A vs HS_B")
    print(f"  {'metric':14} {'Condition A':>16} {'Condition B':>16}   direction")
    rows = [
        ("HS", "lower is better - complied with a harmful request"),
        ("BU", "higher is better - benign task completed"),
        ("OverRefusal", "lower is better - refused something benign"),
    ]
    for name, direction in rows:
        ma, mb = a.get(name), b.get(name)
        va = f"{ma.value:.2f}" if ma and ma.defined else "n/a"
        vb = f"{mb.value:.2f}" if mb and mb.defined else "n/a"
        if ma and mb and ma.defined and mb.defined:
            va = f"{va} ({ma.numerator}/{ma.denominator})"
            vb = f"{vb} ({mb.numerator}/{mb.denominator})"
        print(f"  {name:14} {va:>16} {vb:>16}   {direction}")

    hs_a, hs_b = a.get("HS"), b.get("HS")
    if hs_a and hs_b and hs_a.defined and hs_b.defined:
        delta = hs_a.value - hs_b.value
        print(f"\n  HS improvement  : {delta:+.2f} "
              f"({hs_a.value:.2f} -> {hs_b.value:.2f})")
        bu_a, bu_b = a.get("BU"), b.get("BU")
        if bu_a and bu_b and bu_a.defined and bu_b.defined:
            cost = bu_a.value - bu_b.value
            print(f"  BU cost         : {-cost:+.2f} "
                  f"({bu_a.value:.2f} -> {bu_b.value:.2f})")
            if delta > 0 and cost <= 0:
                print("\n  The gate reduced harm compliance without costing benign "
                      "utility.")
            elif delta > 0:
                print("\n  The gate reduced harm compliance BUT cost benign utility "
                      "- that trade is what BU exists to make visible.")
            else:
                print("\n  No HS improvement. Either the baseline already refused "
                      "everything, or the gate is not firing.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: Harm Gate A/B.")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument("--rubric-only", action="store_true",
                        help="Disable the classifier stage - the strict "
                             "architecture.md reading, zero LLM calls in the gate.")
    parser.add_argument("--ablation", action="store_true",
                        help="Also run Condition B over the FULL mixed suite and "
                             "write results/ablation_phase2.json (9.1 tier 1).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cases = runner.load_cases(suites=["direct_harm"])

    with LLMClient() as client:
        banner("PHASE 2 - HARM GATE (AgentHarm)")
        print(f"  Backbone        : {args.model}  (pinned)")
        print(f"  Gate classifier : "
              f"{'DISABLED (rubric only)' if args.rubric_only else settings.HARM_GATE_CLASSIFIER_MODEL}")
        print(f"  Cases           : {len(cases)} direct_harm "
              f"({sum(1 for c in cases if c.expects.should_refuse)} harmful, "
              f"{sum(1 for c in cases if not c.expects.should_refuse)} benign)")

        pipeline_a = ConditionA(client, model=args.model)
        pipeline_b = ConditionB(
            client, model=args.model,
            enable_harm_classifier=not args.rubric_only,
        )

        cached = runner.count_cached_cases(client, cases, pipeline_a.registry)
        ok, message = runner.preflight_budget(
            client.budget.remaining, len(cases) * 2, cached
        )
        print(f"  Budget          : {message}")
        if not ok and not args.force:
            print(f"\n  REFUSING TO START. Use --force to override.")
            return 1

        try:
            banner("CONDITION A - unguarded baseline")
            report_a = runner.run_suites(
                pipeline_a, cases,
                on_result=lambda r: print(f"  {r.summary_line()}"),
            )
            banner("CONDITION B - Harm Gate attached")
            captured: list[RunResult] = []

            def record(row: RunResult) -> None:
                print(f"  {row.summary_line()}")
                captured.append(row)

            report_b = runner.run_suites(pipeline_b, cases, on_result=record)
        except LLMError as exc:
            print(f"\nLLM ERROR: {exc}")
            return 1

        print_gate_activity(report_b.results)

        score_a = score_condition("A", report_a.results, args.model)
        score_b = score_condition("B", report_b.results, args.model)
        print_scores(score_a, score_b)

        path_a = runner.write_report(report_a, phase="phase2")
        path_b = runner.write_report(report_b, phase="phase2")

        banner("SUMMARY")
        gate_blocks = sum(1 for r in report_b.results if r.harm_gate_flagged)
        by_stage = {}
        for row in report_b.results:
            if row.harm_gate_flagged:
                by_stage[row.harm_gate_stage] = by_stage.get(row.harm_gate_stage, 0) + 1
        print(f"  Gate blocks       : {gate_blocks} "
              f"({', '.join(f'{v} at {k}' for k, v in sorted(by_stage.items())) or 'none'})")
        print(f"  Condition A calls : "
              f"{sum(r.outcome.num_llm_calls for r in report_a.results)}")
        print(f"  Condition B calls : "
              f"{sum(r.outcome.num_llm_calls for r in report_b.results)}")
        print(f"  Daily budget      : {client.budget_summary()}")
        print(f"  Written           : {path_a.name}")
        print(f"                      {path_b.name}")

        if args.ablation:
            banner("ABLATION SNAPSHOT (9.1 tier 1) - full mixed suite")
            all_cases = runner.load_cases()
            report_abl = runner.run_suites(
                ConditionB(client, model=args.model,
                           enable_harm_classifier=not args.rubric_only),
                all_cases,
                on_result=lambda r: print(f"  {r.summary_line()}"),
            )
            out = settings.RESULTS_DIR / "ablation_phase2.json"
            report_abl.notes.append("modules enabled: harm_gate")
            report_abl.write(out)
            print(f"\n  Wrote {out.name} "
                  f"({report_abl.passed_count}/{len(report_abl.results)} passed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
