"""Cumulative ablation table (CLAUDE.md 9.1 tier 1). No LLM calls.

    python demos/ablation_table.py

Renders the saved ablation snapshots as one table: each row is a configuration
of the ensemble, each column a GAI sub-metric. That is the chart Phase 6 turns
into "each paper's contribution is visible and additive".

IT REFUSES TO PRESENT INCOMPARABLE ROWS AS A TREND.

Snapshots are taken at different times as the suite grows, so two rows can
easily be measured over different case sets - which makes any difference
between them partly a change of test, not a change of defense. That is exactly
the mistake a cumulative chart invites: four numbers going down looks like
evidence whether or not the denominators match. When the case sets differ this
prints the overlap and refuses the trend, rather than drawing a line through
points that do not belong on the same axis.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval.schemas import RunResult, load_suites  # noqa: E402
from src.eval.scorer import misalignment_macro_f1  # noqa: E402

RULE = "=" * 84

# Configuration order, matching the phase in which each module was added.
SNAPSHOTS = [
    ("Condition A (no defenses)", "phase1_condition_a_*.json"),
    ("+ Harm Gate", "ablation_phase2.json"),
    ("+ Harm Gate + Planner", "ablation_phase3.json"),
    ("+ Harm Gate + Planner + Firewall/Quarantine", "ablation_phase4.json"),
    ("+ everything (Condition B)", "ablation_phase5.json"),
]


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def rate(n: int, d: int) -> str:
    return f"{n}/{d}={n / d:.2f}" if d else "n/a"


def load(pattern: str) -> dict | None:
    matches = sorted(glob.glob(str(settings.RESULTS_DIR / pattern)))
    if not matches:
        return None
    return json.load(io.open(matches[-1], encoding="utf-8"))


def mf1_for(rows: dict) -> str:
    """MF1 over one snapshot's rows, computed by the scorer, not re-derived.

    Re-implementing a metric for the table is how a chart and a report end up
    disagreeing, so the saved rows are re-validated into `RunResult` and handed
    to the same function Phase 6 will call.

    Snapshots written before Phase 5 carry no checkpoint fields at all; pydantic
    fills the defaults, `misalignment_flagged` stays None, and the metric
    correctly reports n/a rather than inventing a score for a module that did
    not exist when the snapshot was taken.
    """
    results = []
    for row in rows.values():
        try:
            results.append(RunResult.model_validate(
                {**row,
                 "condition": row.get("condition", "B"),
                 "run_index": row.get("run_index", 0),
                 "backbone_model": row.get("backbone_model", ""),
                 "transcript": row.get("transcript", []),
                 "timestamp": row.get("timestamp", "1970-01-01T00:00:00Z")}
            ))
        except Exception:  # noqa: BLE001 - a malformed row must not hide the rest
            continue
    metric = misalignment_macro_f1(results)
    if not metric.defined:
        return "n/a"
    return f"{metric.value:.2f} ({metric.numerator}/{metric.denominator})"


def metrics(rows: dict, cases: dict) -> dict:
    inj = [r for i, r in rows.items() if cases[i].suite == "injection"]
    harm = [
        r for i, r in rows.items()
        if cases[i].suite == "direct_harm" and cases[i].expects.should_refuse
    ]
    benign = [
        i for i in rows if cases[i].expects.should_refuse is False
    ]
    return {
        "passed": sum(1 for r in rows.values() if r["passed"]),
        "n": len(rows),
        "ASR_inj": rate(
            sum(1 for r in inj if r["outcome"]["attack_succeeded"] is True), len(inj)
        ),
        "HS": rate(
            sum(1 for r in harm if not r["outcome"]["refused"]), len(harm)
        ),
        "over_refusal": rate(
            sum(1 for i in benign if rows[i]["outcome"]["refused"]), len(benign)
        ),
        "MF1": mf1_for(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cumulative ablation table.")
    parser.add_argument("--force-trend", action="store_true",
                        help="Print the trend even when case sets differ. Only "
                             "for inspection - never for the report.")
    args = parser.parse_args()

    cases = {c.id: c for c in load_suites(settings.TESTSUITES_DIR)}

    loaded: list[tuple[str, dict, set[str]]] = []
    for label, pattern in SNAPSHOTS:
        data = load(pattern)
        if data is None:
            continue
        rows = {
            r["test_case_id"]: r for r in data["results"]
            if not r.get("error") and r["test_case_id"] in cases
        }
        loaded.append((label, rows, set(rows)))

    if not loaded:
        raise SystemExit("No ablation snapshots in results/. Run the phase demos.")

    banner(f"CUMULATIVE ABLATION - {len(loaded)} configuration(s)")
    print(f"  {'configuration':44} {'cases':>6} {'passed':>8} "
          f"{'ASR_inj':>10} {'HS':>10} {'over-ref':>10} {'MF1':>12}")
    for label, rows, _ in loaded:
        m = metrics(rows, cases)
        print(f"  {label:44} {m['n']:>6} {m['passed']:>4}/{m['n']:<3} "
              f"{m['ASR_inj']:>10} {m['HS']:>10} {m['over_refusal']:>10} "
              f"{m['MF1']:>12}")

    # Comparability gate.
    case_sets = [ids for _, _, ids in loaded]
    common = set.intersection(*case_sets)
    union = set.union(*case_sets)

    banner("COMPARABILITY")
    if len(common) == len(union):
        print(f"  All {len(loaded)} configurations were measured over the same "
              f"{len(common)} cases.")
        print("  The rows are directly comparable and the trend is meaningful.")
        return 0

    print(f"  Case sets DIFFER: union {len(union)}, common to all {len(common)}.")
    for label, _, ids in loaded:
        missing = sorted(union - ids)
        print(f"    {label:44} {len(ids):>3} cases"
              + (f"  missing: {', '.join(missing[:6])}"
                 f"{'...' if len(missing) > 6 else ''}" if missing else ""))

    print()
    print("  REFUSING to present these rows as a trend. A difference between")
    print("  two rows measured over different cases is partly a change of test,")
    print("  not a change of defense - and four numbers going down looks like")
    print("  evidence whether or not the denominators match.")
    print()
    print("  Fix: freeze the suite, then re-run every configuration over it.")
    print("  ConditionB takes enabled_modules, so each row is one command.")

    if args.force_trend:
        banner(f"TREND OVER THE {len(common)} COMMON CASES (inspection only)")
        print(f"  {'configuration':44} {'passed':>8} {'ASR_inj':>10} {'HS':>10}")
        for label, rows, _ in loaded:
            sub = {i: r for i, r in rows.items() if i in common}
            m = metrics(sub, cases)
            print(f"  {label:44} {m['passed']:>4}/{m['n']:<3} "
                  f"{m['ASR_inj']:>10} {m['HS']:>10}")
        print("\n  Still not for the report: the common subset is not the suite")
        print("  any of these configurations was actually evaluated on.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
