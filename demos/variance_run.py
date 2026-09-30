"""Does the headline survive sampling noise? (CLAUDE.md §10, the N-repeat run)

    python demos/variance_run.py --dry-run          # cost first
    python demos/variance_run.py --repeats 3        # the real thing
    python demos/variance_run.py --temperature 0.0  # the determinism check

WHY THIS IS NOT "N=3 AT THE USUAL SETTINGS"
-------------------------------------------
§10 asks for N repeats. Run naively that would measure nothing here, and the
zero it produced would be actively misleading.

`DEFAULT_TEMPERATURE = 0.0`, so the backbone decodes greedily. Measured on
qwen2.5:3b with the cache off, four runs of the same ReAct prompt returned
**byte-identical** replies - 1 distinct output of 4. Repeating the suite at
temperature 0 therefore reports `std = 0.00` for every metric, which reads as
"our result is rock solid" when it actually says "we asked the same question
of a calculator three times".

So the repeats run at a **non-zero temperature**, and that makes this a
robustness check rather than a reproducibility check:

  - the headline numbers stay as they are, at temperature 0, deterministic;
  - this run answers the separate question of whether the *effect* survives
    when the model is allowed to sample.

Both are worth having and they are not the same claim. Reporting the second as
though it were the first would overstate what was measured.

THE CACHE IS OFF, AND IT HAS TO BE
-----------------------------------
CLAUDE.md §5.2.4 is explicit: caching is disabled for eval runs where repeats
are meant to capture real variance. A cached repeat replays the first run
byte-for-byte, so leaving it on would produce a zero for exactly the same
reason temperature 0 does - and far less visibly.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval import report as report_module  # noqa: E402
from src.eval import runner, scorer  # noqa: E402
from src.llm.client import BudgetExceededError, LLMClient, LLMError  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

RULE = "=" * 84

#: Metrics worth a variance estimate. OverRefusal is included because it is the
#: cost side of every safety claim in the report.
TRACKED = ("ASR_inj", "HS", "BU", "UA", "OverRefusal")

#: Measured calls per case, Condition A and the full ensemble (HANDOFF §5.2a).
CALLS_PER_CASE = {"A": 2.8, "B": 11.6}


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def row_path(condition: str, model: str, temperature: float, index: int) -> Path:
    """One repeat's file. Scoped by backbone, temperature AND repeat index.

    All three belong in the name for the same reason the ablation rows carry
    the backbone: a file that silently overwrites another arm's evidence is
    the bug this project has already hit twice.
    """
    slug = runner._slug(model)
    temp = str(temperature).replace(".", "p")
    return settings.RESULTS_DIR / f"variance_{condition}_{slug}_t{temp}_r{index}.json"


def summarise(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean, std = scorer.mean_std(values)
    spread = f"{min(values):.2f}-{max(values):.2f}"
    return f"{mean:.3f} +/- {std:.3f}  (range {spread})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat the A/B run and report variance.")
    parser.add_argument("--model", default=settings.BACKBONE_MODEL)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Non-zero by default: at 0.0 the backbone is "
                             "deterministic and every std is 0 by construction.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Applied globally rather than threaded through every call site: the
    # client reads settings.DEFAULT_TEMPERATURE when it builds each request,
    # so every component - agent loop, planner, judge - moves together. Passing
    # it to only some of them would sample the agent while the judge stayed
    # greedy, which is neither condition.
    settings.DEFAULT_TEMPERATURE = args.temperature

    cases = runner.load_cases()
    banner("VARIANCE RUN - does the headline survive sampling noise?")
    print(f"  Backbone    : {args.model}")
    print(f"  Temperature : {args.temperature}"
          + ("  (deterministic - every std will be 0; see the module docstring)"
             if args.temperature == 0 else ""))
    print(f"  Repeats     : {args.repeats} x Condition A and Condition B")
    print(f"  Cases       : {len(cases)}")
    print("  Cache       : OFF (a cached repeat replays the first one exactly)")

    provider = next(
        (p for p, m in settings.PROVIDER_CHAIN if m == args.model), "groq")
    per_call = (settings.OLLAMA_MEASURED_SECONDS_PER_CALL if provider == "ollama"
                else 60.0 / settings.PROVIDER_LIMITS.get(provider, {}).get(
                    "rate_limit_per_minute", settings.GROQ_RATE_LIMIT_PER_MINUTE))
    calls = sum(CALLS_PER_CASE.values()) * len(cases) * args.repeats
    banner("COST")
    print(f"  ~{calls:.0f} LLM calls at ~{per_call:.1f}s each -> "
          f"~{calls * per_call / 3600:.1f}h")
    print("  Auxiliary safety models stay on their own provider and are rate")
    print("  limited separately, so the real figure runs longer than this.")
    if args.dry_run:
        return 0

    # Cache off is not optional here - see the module docstring.
    with LLMClient(cache_enabled=False) as client:
        for index in range(args.repeats):
            for condition in ("A", "B"):
                path = row_path(condition, args.model, args.temperature, index)
                if path.exists() and not args.force:
                    print(f"\n  [{condition} r{index}] already on disk, skipping "
                          f"({path.name})", flush=True)
                    continue
                banner(f"REPEAT {index + 1}/{args.repeats} - Condition {condition}")
                pipeline = (
                    ConditionA(client, model=args.model) if condition == "A"
                    else ConditionB(client, model=args.model, enabled_modules=None)
                )
                try:
                    report = runner.run_suites(
                        pipeline, cases, run_index=index,
                        on_result=lambda r: print(f"  {r.summary_line()}", flush=True),
                    )
                except BudgetExceededError as exc:
                    print(f"\n  BUDGET EXHAUSTED: {exc}")
                    return 1
                except LLMError as exc:
                    print(f"\n  LLM ERROR: {exc}")
                    return 1
                errored = [r for r in report.results if r.error]
                if len(errored) / max(len(report.results), 1) >= 0.10:
                    print(f"\n  REFUSING to write {path.name}: "
                          f"{len(errored)}/{len(report.results)} cases errored.")
                    return 1
                report.notes.append(
                    f"variance repeat {index} at temperature {args.temperature}, "
                    f"cache disabled"
                )
                report.write(path)
                print(f"\n  Wrote {path.name} "
                      f"({report.passed_count}/{len(report.results)} passed, "
                      f"{len(errored)} errored)")
                print(f"  Budget: {client.budget_summary()}", flush=True)

    report_results(args)
    return 0


def report_results(args) -> None:
    """Mean, standard deviation and range for each metric across the repeats."""
    from src.eval.schemas import SuiteReport

    ids = report_module.refusal_acceptable_ids()
    collected: dict[str, dict[str, list[float]]] = {"A": {}, "B": {}}
    passes: dict[str, list[int]] = {"A": [], "B": []}

    for condition in ("A", "B"):
        for index in range(args.repeats):
            path = row_path(condition, args.model, args.temperature, index)
            if not path.exists():
                continue
            rep = SuiteReport.model_validate_json(path.read_text(encoding="utf-8"))
            score = scorer.score_all(condition, rep.results, rep.backbone_model,
                                     refusal_acceptable_ids=ids)
            passes[condition].append(sum(1 for r in rep.results if r.passed))
            for name in TRACKED:
                metric = score.get(name)
                if metric is not None and metric.defined:
                    collected[condition].setdefault(name, []).append(metric.value)

    banner("VARIANCE ACROSS REPEATS")
    n = min(len(passes["A"]), len(passes["B"]))
    if n < 2:
        print(f"  Only {n} complete repeat(s) on disk - nothing to vary yet.")
        return
    print(f"  {n} repeats, temperature {args.temperature}, cache off.\n")
    print(f"  {'metric':14} {'Condition A':>30} {'Condition B':>30}")
    for name in TRACKED:
        a, b = collected["A"].get(name, []), collected["B"].get(name, [])
        print(f"  {name:14} {summarise(a):>30} {summarise(b):>30}")
    print(f"  {'passed':14} "
          f"{f'{statistics.fmean(passes[chr(65)]):.1f} of 39':>30} "
          f"{f'{statistics.fmean(passes[chr(66)]):.1f} of 39':>30}")

    banner("DOES THE EFFECT SURVIVE?")
    print("  The question is whether Condition B beats Condition A in EVERY")
    print("  repeat, not whether the means differ - a mean can hide one repeat")
    print("  where the defense lost.\n")
    for name in ("ASR_inj", "HS"):
        a, b = collected["A"].get(name, []), collected["B"].get(name, [])
        if len(a) < 2 or len(b) < 2:
            continue
        pairs = list(zip(a, b))
        always = all(bv <= av for av, bv in pairs)
        print(f"  {name:10} B <= A in {sum(1 for av, bv in pairs if bv <= av)}"
              f"/{len(pairs)} repeats"
              f"   {'- holds every time' if always else '- DOES NOT always hold'}")
    if args.temperature == 0:
        print()
        print("  Temperature is 0, so these repeats are deterministic and the")
        print("  spreads above are all zero by construction. That is a")
        print("  reproducibility check, NOT a robustness one.")


if __name__ == "__main__":
    raise SystemExit(main())
