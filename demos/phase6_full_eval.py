"""Phase 6 - the GAI composite and the final report (CLAUDE.md §10).

    python demos/phase6_full_eval.py              # build results/report.md
    python demos/phase6_full_eval.py --dry-run    # what it would cost, spend nothing
    python demos/phase6_full_eval.py --timing 4   # the uncached LAT measurement
    python demos/phase6_full_eval.py --isolation  # §9.1 tier 2 single-module rows

WHY THE DEFAULT SPENDS NOTHING
------------------------------
§10's definition of done is "every suite x both conditions x N repeats". That
run already exists: `demos/frozen_ablation.py` measured all five cumulative
configurations - which includes both Condition A and Condition B - over one
frozen 39-case set, in one sitting, on the pinned backbone, on 2026-09-19.
Re-running it would replay from cache and reproduce byte-identical numbers at
roughly eight hours of wall clock, because the binding constraint is Groq's
2 requests/minute and not the daily cap (HANDOFF §5.2a).

So the default rebuilds the report *from those snapshots* and names the source
file behind every row, rather than re-running the agent to arrive at the same
place. `--rerun` exists if the suite has changed underneath them; it delegates
to `frozen_ablation.py` rather than growing a second copy of that logic.

**N = 1, and that is a scoping decision, not an oversight.** §5.4 explicitly
permits it with a documented note. At 2 requests/minute, N=3 over 39 cases x
5 configurations is roughly 24 hours of wall clock with a cold cache. The
report says so in its own limitations section; the consequence is that no
sub-metric here carries a run-to-run variance estimate, only a binomial
interval over cases.

WHAT STILL COSTS REQUESTS
-------------------------
`--timing` is the one measurement that cannot come off disk. LAT is wall
clock, and every saved result in this project replays from a cache that
returns in ~17ms, so `scorer.added_latency` refuses to score them (it checks
recorded cache provenance rather than trusting the timestamps). The timing
pass therefore runs with the cache switched off entirely - which also means it
writes nothing back, so it cannot perturb any other demo's replay.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval import report as report_module  # noqa: E402
from src.eval import runner  # noqa: E402
from src.llm.client import BudgetExceededError, LLMClient, LLMError  # noqa: E402
from demos import frozen_ablation  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

RULE = "=" * 84

# Single-module isolation (§9.1 tier 2). Harm-Gate-only is already the
# cumulative ablation's second row, so it is not repeated here.
ISOLATION_ROWS: list[tuple[str, str, set[str]]] = [
    ("Planner only", "isolation_planner", {"planner"}),
    ("Firewall + Quarantine only", "isolation_firewall", {"firewall", "quarantine"}),
    ("Misalignment only", "isolation_misalignment", {"misalignment"}),
]


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def timing_run(client: LLMClient, count: int) -> int:
    """Measure LAT: the same cases through both conditions, cache off.

    Deliberately small. LAT is a per-tool-call ratio, so it does not need the
    whole suite to be estimated - it needs *some* cases that genuinely make
    tool calls in both arms, run fresh. Condition A costs ~2.8 calls per case
    and Condition B ~11.6 (HANDOFF §5.2a), so four cases is roughly 58
    requests and half an hour at 2 requests/minute.

    Injection-suite workspace cases, because they are the ones that exercise
    the full defended path - plan enforcement, response scanning, quarantine.
    A benign case that trips no module would measure the overhead of the
    modules not firing, which is the flattering half of the question.
    """
    cases = [
        c for c in runner.load_cases()
        if c.scenario == "workspace" and c.suite == "injection"
    ][:count]
    if not cases:
        print("  No workspace injection cases found; nothing to time.")
        return 1

    print(f"  Timing {len(cases)} case(s), cache OFF: "
          + ", ".join(c.id for c in cases))
    print(f"  Expect roughly {len(cases) * (2.8 + 11.6):.0f} requests, "
          f"{len(cases) * (2.8 + 11.6) / settings.GROQ_RATE_LIMIT_PER_MINUTE / 60:.1f}h "
          f"at {settings.GROQ_RATE_LIMIT_PER_MINUTE}/min.\n")

    for label, stem, pipeline in (
        ("A (bare backbone)", "phase6_timing_a", ConditionA(client)),
        ("B (full ensemble)", "phase6_timing_b",
         ConditionB(client, enabled_modules=None)),
    ):
        path = settings.RESULTS_DIR / f"{stem}.json"
        if path.exists():
            print(f"  {label}: already on disk ({path.name}), skipping.")
            continue
        banner(f"TIMING - Condition {label}")
        started = time.monotonic()
        report = runner.run_suites(
            pipeline, cases,
            on_result=lambda r: print(
                f"  {r.test_case_id}: {r.outcome.latency_ms}ms over "
                f"{r.outcome.num_tool_calls} tool call(s), "
                f"{r.outcome.cache_hits} cache hit(s)", flush=True),
        )
        report.notes.append(
            "LAT timing run: cache disabled, so these latencies are wall clock"
        )
        report.write(path)
        print(f"  Wrote {path.name} in {time.monotonic() - started:.0f}s")
        print(f"  Budget: {client.budget_summary()}", flush=True)
    return 0


def isolation_run(client: LLMClient, force: bool, model: str = "") -> int:
    """§9.1 tier 2: each module alone against the full frozen suite.

    The claim this supports is the one CLAUDE.md §9.1 says the project *can*
    honestly make: each module scores well on its own sub-metric and
    near-baseline on the others, and only the ensemble is good across all of
    them. The cumulative ablation cannot show that - in it, every row
    contains every earlier module.
    """
    cases = runner.load_cases()
    model = model or settings.BACKBONE_MODEL
    for label, stem, modules in ISOLATION_ROWS:
        # Backbone-scoped, for the same reason frozen_ablation.row_path is:
        # a second arm must not overwrite the first arm's evidence.
        path = frozen_ablation.row_path(stem, model)
        if path.exists() and not force:
            print(f"  {label}: already on disk ({path.name}), skipping.")
            continue
        banner(f"ISOLATION - {label}")
        # The model must be passed through, or the isolation rows silently
        # measure a different backbone from the ablation they are compared
        # against - the confound this project keeps guarding against.
        pipeline = ConditionB(client, model=model, enabled_modules=modules)
        report = runner.run_suites(
            pipeline, cases,
            on_result=lambda r: print(f"  {r.summary_line()}", flush=True),
        )
        report.notes.append(f"single-module isolation: {', '.join(sorted(modules))}")
        errored = [r for r in report.results if r.error]
        if not frozen_ablation.accept_row(label, path, errored, len(report.results)):
            return 1
        note = frozen_ablation.module_health_note(report.results)
        if not frozen_ablation.accept_module_health(path, report.results, accept=True):
            pass  # accepted, but the note below records it
        if note:
            report.notes.append(note)
        report.write(path)
        print(f"  Wrote {path.name} ({report.passed_count}/{len(report.results)})")
        print(f"  Budget: {client.budget_summary()}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: GAI + final report.")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL,
                        help="Backbone for the timing/isolation runs. "
                             "Rows are written under a backbone-scoped "
                             "filename so arms cannot overwrite each other.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Say what each mode would cost; spend nothing.")
    parser.add_argument("--timing", type=int, metavar="N", default=0,
                        help="Run the uncached LAT measurement over N cases.")
    parser.add_argument("--isolation", action="store_true",
                        help="Run the §9.1 tier 2 single-module rows.")
    parser.add_argument("--rerun", action="store_true",
                        help="Regenerate the frozen ablation snapshots first.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run rows already on disk.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    banner("PHASE 6 - GUARDED AGENT INDEX AND FINAL REPORT")
    print(f"  Backbone : {settings.BACKBONE_MODEL} (pinned)")
    print(f"  Results  : {settings.RESULTS_DIR}")

    if args.dry_run:
        # Seconds per call for the arm actually selected, not a quota. See
        # frozen_ablation.estimate: converting a requests/minute quota into
        # wall clock prices a local run at "0.0h" and a hosted one at ~10h.
        provider = next(
            (prov for prov, name in settings.PROVIDER_CHAIN if name == args.model),
            "groq",
        )
        if provider == "ollama":
            per_call = settings.OLLAMA_MEASURED_SECONDS_PER_CALL
            basis = f"~{per_call:.1f}s/call measured locally; no provider quota"
        else:
            rate = settings.PROVIDER_LIMITS.get(provider, {}).get(
                "rate_limit_per_minute", settings.GROQ_RATE_LIMIT_PER_MINUTE)
            per_call = 60.0 / rate
            basis = f"{rate} requests/minute, so {per_call:.0f}s/call"
        cases = len(runner.load_cases())
        banner("COST, IF EACH MODE WERE RUN")
        print(f"  {'mode':34} {'est. calls':>11} {'est. time':>10}")
        print(f"  {'report only (the default)':34} {0:>11} {'instant':>10}")
        for n in (4,):
            calls = n * (2.8 + 11.6)
            print(f"  {f'--timing {n}':34} {calls:>11.0f} "
                  f"{calls * per_call / 3600:>9.1f}h")
        iso = cases * (7.1 + 8.9 + 11.6)
        print(f"  {'--isolation (3 rows)':34} {iso:>11.0f} "
              f"{iso * per_call / 3600:>9.1f}h")
        print(f"\n  Backbone {args.model} via {provider}: {basis}.")
        print("  Cached cases cost nothing; the isolation figure ignores the "
              "cache and is an upper bound.")
        return 0

    if args.rerun:
        from demos import frozen_ablation
        code = frozen_ablation.main()
        if code:
            return code

    try:
        # Two clients, because they want opposite cache settings and mixing
        # them would either fabricate LAT or burn eight hours on the isolation
        # rows the cache can serve for free.
        if args.timing:
            with LLMClient(cache_enabled=False) as client:
                code = timing_run(client, args.timing)
                if code:
                    return code
        if args.isolation:
            with LLMClient() as client:
                code = isolation_run(client, args.force, args.model)
                if code:
                    return code
    except BudgetExceededError as exc:
        print(f"\n  BUDGET EXHAUSTED: {exc}")
        return 1
    except LLMError as exc:
        print(f"\n  LLM ERROR: {exc}")
        return 1

    banner("BUILDING THE REPORT")
    path = report_module.build_report()
    print(f"\n  Wrote {path}")
    print("  This is the Phase 6 deliverable (§10).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
