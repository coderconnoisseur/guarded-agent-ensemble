"""Compare saved Condition A runs across backbones. No LLM calls.

    python demos/compare_backbones.py

Reads every `results/phase*_condition_a_*.json` file and prints a per-case
matrix, per-arm sub-metrics, and the pairwise overlap between each backbone's
set of failures.

The overlap table exists to keep a specific claim honest. "The backbones fail
on disjoint cases" is easy to assert from memory and easy to get wrong - it
was, once - so the numbers behind it are computed here from the saved runs
instead, restricted to the cases both backbones actually completed.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval.schemas import load_suites  # noqa: E402

RULE = "=" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def load_runs(pattern: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(pattern)):
        data = json.load(io.open(path, encoding="utf-8"))
        runs[data["backbone_model"]] = {
            r["test_case_id"]: r for r in data["results"]
        }
    return runs


def completed(run: dict[str, dict]) -> set[str]:
    """Cases with real data. A quota failure is absence of evidence."""
    return {cid for cid, r in run.items() if not r.get("error")}


def failed(run: dict[str, dict]) -> set[str]:
    return {
        cid for cid, r in run.items() if not r.get("error") and not r["passed"]
    }


def print_matrix(runs: dict[str, dict[str, dict]], cases: dict) -> None:
    banner("PER-CASE VERDICTS")
    names = list(runs)
    width = 22
    print(f"  {'case':12} {'arm':22} " + " ".join(f"{n[:width]:>{width}}" for n in names))
    for cid in sorted(cases):
        arm = cases[cid].category[:22]
        cells = []
        for name in names:
            row = runs[name].get(cid)
            if row is None:
                cells.append("-")
            elif row.get("error"):
                cells.append("no data")
            else:
                cells.append("pass" if row["passed"] else "FAIL")
        print(f"  {cid:12} {arm:22} " + " ".join(f"{c:>{width}}" for c in cells))


def print_metrics(runs: dict[str, dict[str, dict]], cases: dict) -> None:
    banner("SUB-METRICS BY ATTACK ARM")

    arms = {
        "ASR_inj blunt": (
            {i for i, c in cases.items()
             if c.suite == "injection" and "delegated" not in c.category},
            "attack_succeeded", True,
        ),
        "ASR_inj delegated": (
            {i for i, c in cases.items()
             if c.suite == "injection" and "delegated" in c.category},
            "attack_succeeded", True,
        ),
        "HS plain": (
            {i for i, c in cases.items()
             if c.suite == "direct_harm" and i.startswith("harm")
             and "jailbreak" not in c.category},
            "refused", False,
        ),
        "HS jailbreak": (
            {i for i, c in cases.items() if "jailbreak" in c.category},
            "refused", False,
        ),
        "BU benign": (
            {i for i in cases if i.startswith("benign")}, "task_completed", True,
        ),
    }

    names = list(runs)
    print(f"  {'metric':20} " + " ".join(f"{n[:22]:>22}" for n in names))
    for label, (ids, field, want) in arms.items():
        cells = []
        for name in names:
            rows = [
                runs[name][i] for i in ids
                if i in runs[name] and not runs[name][i].get("error")
            ]
            if not rows:
                cells.append("no data")
            else:
                hits = sum(1 for r in rows if r["outcome"][field] is want)
                cells.append(f"{hits}/{len(rows)} = {hits/len(rows):.2f}")
        print(f"  {label:20} " + " ".join(f"{c:>22}" for c in cells))


def print_overlap(runs: dict[str, dict[str, dict]]) -> None:
    banner("FAILURE OVERLAP (only cases both backbones completed)")
    if len(runs) < 2:
        print("  Need at least two runs to compare.")
        return

    for a, b in combinations(runs, 2):
        shared = completed(runs[a]) & completed(runs[b])
        fa, fb = failed(runs[a]) & shared, failed(runs[b]) & shared
        both, either = fa & fb, fa | fb
        jaccard = len(both) / len(either) if either else 0.0

        print(f"\n  {a}")
        print(f"  {b}")
        print(f"    cases compared    : {len(shared)}")
        print(f"    failures, first   : {len(fa)}  {sorted(fa) or '-'}")
        print(f"    failures, second  : {len(fb)}  {sorted(fb) or '-'}")
        print(f"    shared failures   : {len(both)}  {sorted(both) or '-'}")
        print(f"    overlap (Jaccard) : {jaccard:.2f}")
        if not both and either:
            verdict = "DISJOINT - no case fails on both"
        elif fa and fb and (fa <= fb or fb <= fa):
            verdict = "NESTED - one backbone's failures are a subset of the other's"
        elif jaccard >= 0.5:
            verdict = "LARGELY SHARED"
        else:
            verdict = "PARTIALLY OVERLAPPING"
        print(f"    verdict           : {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare backbones on saved runs.")
    parser.add_argument(
        "--pattern",
        default=str(settings.RESULTS_DIR / "phase*_condition_a_*.json"),
        help="Glob for result files to compare.",
    )
    args = parser.parse_args()

    runs = load_runs(args.pattern)
    if not runs:
        raise SystemExit(
            f"No results matched {args.pattern}. Run demos/phase1_demo.py first."
        )
    cases = {c.id: c for c in load_suites(settings.TESTSUITES_DIR)}

    banner(f"CONDITION A ACROSS {len(runs)} BACKBONE(S)")
    for name, run in runs.items():
        done, fails = completed(run), failed(run)
        print(f"  {name:34} {len(done)}/{len(run)} completed, {len(fails)} failed")

    print_matrix(runs, cases)
    print_metrics(runs, cases)
    print_overlap(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
