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



# A row with this share of errored cases or more is not written at all.
#
# MEASURED, the hard way: the local Ollama server crashed partway through a
# Condition A run and 15 of 39 cases errored with ConnectError, 14 of them
# before making a single LLM call. The row was written anyway, reported
# "12/39 passed", and - because a row already on disk is skipped without
# --force - a resumed run would have treated that as a finished measurement
# and built a five-row trend on top of it.
#
# An errored case is not a failed case. It carries no evidence at all, so it
# silently shrinks every denominator the scorer computes: `passed` drops,
# ASR_inj and HS are taken over whatever survived, and nothing on the table
# says the arm lost a third of its cases. That is precisely the silent-number
# class this project keeps finding (HANDOFF §1).
#
# 10% rather than zero, because a single transient provider hiccup in 39 cases
# is noise; a third of the suite is an outage.
MAX_ERROR_SHARE = 0.10


def accept_row(label: str, path: Path, errored: list, total: int) -> bool:
    """Whether this row is a measurement or the wreckage of an outage."""
    if not errored:
        return True

    share = len(errored) / total if total else 1.0
    kinds: dict[str, int] = {}
    for result in errored:
        key = (result.error or "").split(":")[0][:60]
        kinds[key] = kinds.get(key, 0) + 1
    summary = "; ".join(f"{count}x {kind}" for kind, count in
                        sorted(kinds.items(), key=lambda kv: -kv[1]))

    if share < MAX_ERROR_SHARE:
        print(f"\n  NOTE: {len(errored)}/{total} case(s) errored "
              f"({share:.0%}) - under the {MAX_ERROR_SHARE:.0%} threshold, so "
              f"the row is kept. {summary}")
        return True

    print(f"\n  REFUSING TO WRITE {path.name}")
    print(f"  {len(errored)}/{total} cases errored ({share:.0%}), over the "
          f"{MAX_ERROR_SHARE:.0%} threshold.")
    print(f"  {summary}")
    print()
    print("  An errored case carries no evidence, so writing this row would")
    print("  shrink every denominator in the table without saying so - and a")
    print("  row on disk is skipped on the next run, so it would be treated as")
    print("  a finished measurement. Fix the cause and re-run this row.")
    return False



# A module that failed open on this share of its runs or more invalidates the
# row, for the same reason an outage does.
#
# MEASURED, and it is the sharper version of the errored-case problem: the
# Misalignment Checkpoint was handed model="qwen2.5:3b" while keeping the
# configured provider "groq", which has no such model. Every judge call 404'd
# into an LLMError and the checkpoint took its documented fail-open path 15
# times in one 39-case run.
#
# Nothing looked wrong. `misalignment_ran` was True, no case errored, the row
# wrote cleanly, and the ablation table reported that adding the checkpoint
# changed nothing - which read as a finding about the module rather than a
# finding about a typo. Failing open on a provider outage is deliberate
# (HANDOFF §4); a row where it happened on 15 of 39 runs is not a measurement
# OF that module, and the two have to be distinguishable from the outside.
MAX_DEGRADED_SHARE = 0.10

# (field that marks a degraded run, field that marks the module as active)
_MODULE_HEALTH = [
    ("Planner", "plan_degraded", "plan_ran"),
    ("Misalignment Checkpoint", "misalignment_degraded", "misalignment_ran"),
]


def module_health_note(results: list) -> str:
    """A note recording which modules failed open, for the row's own record.

    Written into the snapshot rather than only printed, so a reader months
    later can tell "this module changed nothing" from "this module never ran".
    """
    parts = []
    for name, degraded_field, ran_field in _MODULE_HEALTH:
        active = [r for r in results if getattr(r, ran_field, False)]
        degraded = [r for r in active if getattr(r, degraded_field, False)]
        if degraded:
            parts.append(f"{name} failed open on {len(degraded)}/{len(active)} runs")
    return ("DEGRADED: " + "; ".join(parts)) if parts else ""


def accept_module_health(path: Path, results: list, accept: bool = False) -> bool:
    """Whether every enabled defense actually ran, or quietly failed open.

    `accept` is for the case where the degradation IS the finding - a
    backbone too weak to emit a valid plan a quarter of the time is a real
    property of that arm, not a fault to fix. It still prints, and the
    caller stamps it into the row's notes, so the row can never be read
    later as though the module had worked.
    """
    ok = True
    for name, degraded_field, ran_field in _MODULE_HEALTH:
        active = [r for r in results if getattr(r, ran_field, False)]
        if not active:
            continue
        degraded = [r for r in active if getattr(r, degraded_field, False)]
        if not degraded:
            continue
        share = len(degraded) / len(active)
        if share < MAX_DEGRADED_SHARE:
            print(f"\n  NOTE: {name} failed open on {len(degraded)}/"
                  f"{len(active)} run(s) ({share:.0%}) - under the "
                  f"{MAX_DEGRADED_SHARE:.0%} threshold, row kept.")
            continue
        ok = False
        print(f"\n  REFUSING TO WRITE {path.name}")
        print(f"  {name} failed open on {len(degraded)}/{len(active)} runs "
              f"({share:.0%}), over the {MAX_DEGRADED_SHARE:.0%} threshold.")
        print()
        print("  The module was wired in and recorded as running, but it took")
        print("  its fail-open path instead of ruling. A row like this reads as")
        print("  'the module changed nothing', which is a claim about the")
        print("  module rather than about whatever broke it. Usual cause: the")
        print("  judge model and its provider disagree - see")
        print("  MisalignmentCheckpoint.__init__.")
    return True if accept else ok


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def row_path(stem: str, model: str = "") -> Path:
    """Where one row is written, scoped by backbone when it is not the pinned one.

    THE FILENAME MUST CARRY THE MODEL OR THIS COMMAND DESTROYS ITS OWN
    EVIDENCE. These stems used to be fixed, so running the ablation against a
    second backbone would overwrite the pinned-backbone snapshots that the
    whole Phase 6 report is computed from - and `--force` would do it
    silently, in five files, with no way back short of a ~10h re-run.

    That is not hypothetical: the same class of bug already overwrote the
    Condition A baseline twice (HANDOFF §1), which is why
    `runner.default_report_name()` appends the scope of a narrowed run. This
    is the same rule applied to the backbone.

    The pinned model keeps the bare name, so existing files and
    `ablation_table.py`'s patterns are untouched.
    """
    if model and model != settings.BACKBONE_MODEL:
        return settings.RESULTS_DIR / f"{stem}_{runner._slug(model)}.json"
    return settings.RESULTS_DIR / f"{stem}.json"


def estimate(cases: list[TestCase], client: LLMClient, model: str = "") -> None:
    """Print what this will cost before spending any of it."""
    from src.tools.registry import build_registry

    by_scenario: dict[str, list[TestCase]] = {}
    for case in cases:
        by_scenario.setdefault(case.scenario, []).append(case)
    cached = sum(
        runner.count_cached_cases(client, group, build_registry(scenario))
        for scenario, group in by_scenario.items()
    )

    # SECONDS PER CALL, not a rate limit, because the two arms are bound by
    # completely different things and using one number for both is how a
    # ~1.2h job got priced at "0.0h".
    #
    #   hosted: a request every 30s because the limiter says so (2/min), and
    #           the model's own latency disappears inside that wait.
    #   local:  no limiter at all - the ceiling is this machine's
    #           tokens/second, measured at 3.3s/call for qwen2.5:3b.
    #
    # Quoting Groq's 2/min for a local run would price it at ~10h; quoting
    # Ollama's nominal 600/min would price it at 0.0h. Both are wrong in the
    # direction that changes the decision.
    provider = next(
        (prov for prov, name in settings.PROVIDER_CHAIN if name == model), "groq"
    )
    if provider == "ollama":
        seconds_per_call = settings.OLLAMA_MEASURED_SECONDS_PER_CALL
        basis = (f"~{seconds_per_call:.1f}s/call measured on this machine; "
                 f"no provider quota")
    else:
        rate = settings.PROVIDER_LIMITS.get(provider, {}).get(
            "rate_limit_per_minute", settings.GROQ_RATE_LIMIT_PER_MINUTE
        )
        seconds_per_call = 60.0 / rate
        basis = (f"{rate} requests/minute, so {seconds_per_call:.0f}s/call - the "
                 f"rate limit binds, not the {settings.GROQ_DAILY_REQUEST_CAP}/day cap")

    total = 0.0
    print(f"  {len(cases)} frozen cases, {cached} starting from cache")
    print(f"\n  {'row':46} {'est. calls':>11} {'est. time':>10}")
    for (label, stem, _), per_case in zip(ROWS, MEASURED_CALLS_PER_CASE):
        if row_path(stem, model).exists():
            print(f"  {label:46} {'(done)':>11} {'-':>10}")
            continue
        calls = len(cases) * per_case
        total += calls
        print(f"  {label:46} {calls:>11.0f} "
              f"{calls * seconds_per_call / 3600:>9.1f}h")
    print(f"  {'TOTAL (upper bound, ignores cache)':46} "
          f"{total:>11.0f} {total * seconds_per_call / 3600:>9.1f}h")
    print(f"\n  Backbone {model or settings.BACKBONE_MODEL} via {provider}: {basis}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-suite cumulative ablation.")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cost estimate and exit.")
    parser.add_argument("--accept-degraded", action="store_true",
                        help="Write a row even when a module failed open on "
                             "many runs. For when the degradation is a "
                             "measured property of the backbone rather than "
                             "a fault; it is stamped into the row notes.")
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
        estimate(cases, client, args.model)
        if args.dry_run:
            return 0

        for index, (label, stem, modules) in enumerate(ROWS, start=1):
            path = row_path(stem, args.model)
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

            errored = [r for r in report.results if r.error]
            if not accept_row(label, path, errored, len(report.results)):
                return 1
            degraded_note = module_health_note(report.results)
            if not accept_module_health(path, report.results,
                                        accept=args.accept_degraded):
                return 1
            if degraded_note:
                report.notes.append(degraded_note)

            report.write(path)
            print(f"\n  Wrote {path.name} "
                  f"({report.passed_count}/{len(report.results)} passed, "
                  f"{len(errored)} errored)")
            print(f"  Budget: {client.budget_summary()}", flush=True)

        banner("DONE")
        print("  All five rows are over the same frozen case set, so")
        print("  `python demos/ablation_table.py` should now accept the trend.")
        print(f"  Budget: {client.budget_summary()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
