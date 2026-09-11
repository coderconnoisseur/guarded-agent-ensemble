"""Runs one condition against one or more test suites (CLAUDE.md 10, Phase 1).

Responsibilities, in order, for each case:

  1. Reset the environment - sandbox files, mock outbox, web stub corpus,
     inbox. Without this, a case that deletes every file or sends an email
     changes what the *next* case sees, and a failure gets attributed to the
     wrong test.
  2. Plant the case's injected payload, if it has one, into whichever tool
     response is supposed to carry it.
  3. Run the task through the condition's pipeline.
  4. Grade the result mechanically (see `grading.py` - no LLM judge).
  5. Record an 8.2-shaped RunResult.

Note on the `diversity` suite: `src/eval/testsuites/diversity/` is
deliberately empty and Phase 1 does not populate it. DIV_ASR is meant to score
a *generated* corpus from an AgentVigil-style mutation loop (3, 11), so
hand-writing cases would misrepresent what that number measures. 9 already
prescribes the honest handling - document the omission and either renormalize
the remaining weights or use a flagged placeholder. See the README in that
folder. This is a scoping decision, not an oversight.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from config import settings
from src.agent.loop import AgentResult
from src.eval.grading import grade
from src.eval.schemas import RunResult, SuiteReport, TestCase, load_suites
from src.tools import comms, files, web

logger = logging.getLogger(__name__)

# Conservative per-case estimate for the budget preflight: a ReAct loop
# typically spends 2-4 calls on these cases (5.4 says 3-6 for richer ones).
ESTIMATED_CALLS_PER_CASE = 4


class Pipeline(Protocol):
    """What the runner needs from a condition. ConditionA and, from Phase 2,
    ConditionB both satisfy this."""

    condition: str

    def run(self, task: str) -> AgentResult: ...


def reset_environment() -> None:
    """Return every stateful tool to its seed state."""
    files.reset_sandbox()
    comms.clear_outbox()
    comms.reset_inbox()
    web.reset_pages()


def plant_injection(case: TestCase) -> None:
    """Load a case's hidden payload into the tool response that carries it.

    Phase 1 only needs this for the injection suite, but the mechanism is
    general: the payload is data the *environment* returns, never part of the
    user's prompt, which is exactly what makes it an indirect injection.
    """
    if not case.injected_content:
        return

    tool = case.injection_tool
    if tool == "web.fetch":
        if not case.injection_key:
            raise ValueError(f"{case.id}: web.fetch injection needs an injection_key")
        web.register_page(case.injection_key, case.injected_content)
    elif tool == "files.read":
        if not case.injection_key:
            raise ValueError(f"{case.id}: files.read injection needs an injection_key")
        files.write_file(case.injection_key, case.injected_content)
    elif tool == "comms.list_inbox":
        comms.add_inbox_message(
            case.injection_key or "unknown@external.example",
            "Re: your request",
            case.injected_content,
        )
    else:
        raise ValueError(f"{case.id}: cannot plant an injection into {tool!r}")

    verify_injection(case)
    logger.debug("Planted injection for %s into %s", case.id, tool)


def verify_injection(case: TestCase) -> None:
    """Read the payload back through the tool that is supposed to carry it.

    A payload that fails to land turns an injection case into a no-op: the
    agent has nothing to be attacked by, the case passes, and ASR_inj is
    quietly deflated. Nothing in the transcript says "the attack was never
    delivered" - it just looks like the defense worked.

    This happened once in a live Phase 4 ablation and could not be reproduced
    offline, so rather than trust a cause we have not found, the delivery is
    now asserted. A silent no-op becomes a loud failure.
    """
    if not case.injected_content:
        return

    marker = next(
        (line.strip() for line in case.injected_content.splitlines() if line.strip()),
        "",
    )[:40]

    if case.injection_tool == "web.fetch":
        seen = web.fetch(case.injection_key or "")
    elif case.injection_tool == "files.read":
        seen = files.read_file(case.injection_key or "")
    elif case.injection_tool == "comms.list_inbox":
        seen = comms.list_inbox()
    else:
        return

    if marker and marker not in seen:
        raise RuntimeError(
            f"{case.id}: injection payload did not land in {case.injection_tool} "
            f"(key={case.injection_key!r}). The case would have run with no "
            f"attack present and silently deflated ASR_inj. Refusing to "
            f"continue. Saw: {seen[:120]!r}"
        )


def count_cached_cases(client, cases: list[TestCase], registry) -> int:
    """How many cases will replay entirely from disk, costing nothing.

    Checks whether each case's *first* LLM call is already cached. At
    temperature 0 the whole trajectory is deterministic, so a cached first
    call means the rest of that case's chain replays identically from disk
    too - which is what makes a demo re-runnable in front of an audience
    without spending a single request.

    This mirrors how `ReActAgent` builds its opening messages. If that
    construction ever changes, this count drifts low and the estimate simply
    becomes conservative again, which is the safe direction to be wrong in.
    """
    from src.agent import prompts
    from src.llm.client import DiskCache

    system = prompts.build_system_prompt(registry.describe_for_prompt())
    params = {
        "temperature": settings.DEFAULT_TEMPERATURE,
        "max_tokens": settings.DEFAULT_MAX_TOKENS,
    }
    cached = 0
    for case in cases:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": case.prompt},
        ]
        key = DiskCache.make_key(client.current_model, messages, params)
        if client.cache.get(key) is not None:
            cached += 1
    return cached


def capture_world_state() -> dict[str, str]:
    """Snapshot every readable sandbox file, path -> contents.

    Taken immediately after a run and stored on the result, so checks about
    what survived can be evaluated later from the saved JSON alone.
    """
    root = settings.SANDBOX_DIR
    snapshot: dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                snapshot[path.relative_to(root).as_posix()] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                logger.debug("Could not snapshot %s", path)
    return snapshot


def preflight_budget(
    remaining: int, num_cases: int, cached: int = 0
) -> tuple[bool, str]:
    """Would this run fit in what is left of today's request budget?

    Returns (ok, message). The caller decides whether to honour a refusal -
    the demo exposes --force - but the estimate is printed either way so a run
    never silently walks into a BudgetExceededError halfway through a suite.

    Cached cases are excluded from the estimate: counting them would refuse a
    replay that actually costs nothing.
    """
    fresh = max(0, num_cases - cached)
    estimate = fresh * ESTIMATED_CALLS_PER_CASE

    if cached and not fresh:
        return True, (
            f"{num_cases} case(s), all already cached - this run will not spend "
            f"any requests ({remaining} left today)"
        )

    message = (
        f"{num_cases} case(s), {cached} cached, {fresh} fresh; roughly "
        f"{estimate} LLM calls at {ESTIMATED_CALLS_PER_CASE}/case; "
        f"{remaining} request(s) left today"
    )
    if estimate > remaining:
        return False, (
            f"{message}. This run may not fit. Re-run with --force to try anyway, "
            f"or narrow it with --suite / --limit. Cached cases cost nothing."
        )
    return True, message


def run_case(
    pipeline: Pipeline, case: TestCase, run_index: int = 0
) -> RunResult:
    """Run and grade a single test case in a freshly reset environment."""
    reset_environment()
    plant_injection(case)

    logger.info("[%s] %s: %.70s", pipeline.condition, case.id, case.prompt)
    result = pipeline.run(case.prompt)
    world = capture_world_state()
    outcome, check_results, passed, reason = grade(case, result, world)

    # Defense modules attach their verdict to the AgentResult; the runner
    # stays ignorant of what they are and just records what it finds.
    verdict = getattr(result, "harm_gate_verdict", None)
    enforcement = getattr(result, "plan_enforcement", None)
    verdicts = getattr(result, "firewall_verdicts", None) or []
    quarantines = getattr(result, "quarantine_events", None) or []
    tom = getattr(result, "misalignment_verdicts", None)

    return RunResult(
        test_case_id=case.id,
        condition=pipeline.condition,
        run_index=run_index,
        backbone_model=", ".join(result.models_used) or "(none)",
        transcript=result.transcript,
        outcome=outcome,
        timestamp=datetime.now(timezone.utc).isoformat(),
        suite=case.suite,
        category=case.category,
        expects_refusal=bool(case.expects.should_refuse),
        harm_gate_flagged=bool(verdict and verdict.flagged),
        harm_gate_stage=(verdict.stage if verdict else ""),
        harm_gate_category=(verdict.category if verdict else None),
        plan_ran=enforcement is not None,
        plan_node_count=(len(enforcement.graph.nodes) if enforcement else 0),
        plan_executed=(list(enforcement.executed) if enforcement else []),
        plan_expansions=(list(enforcement.expansions) if enforcement else []),
        plan_rejections=([t for t, _ in enforcement.rejections] if enforcement else []),
        plan_degraded=(enforcement.graph.degraded if enforcement else False),
        misalignment_expected=case.expects.misalignment_expected,
        checkpoint_label=case.expects.checkpoint_label,
        # `misalignment_ran` distinguishes "the module was off" from "it was on
        # but no critical action reached it" - both leave the check count at 0,
        # and confusing them would make a Planner block look like a checkpoint
        # that found nothing to object to.
        misalignment_ran=tom is not None,
        misalignment_checks=sum(1 for v in (tom or []) if not v.degraded),
        misalignment_degraded=any(v.degraded for v in (tom or [])),
        misalignment_reasons=[v.reason for v in (tom or []) if v.flagged],
        misalignment_inferred_tasks=[v.inferred_task for v in (tom or [])],
        firewall_flagged=any(v.flagged for v in verdicts),
        firewall_stages=[v.stage for v in verdicts if v.flagged],
        firewall_signals=sorted({s for v in verdicts for s in v.signals}),
        quarantined_tools=[e.tool for e in quarantines],
        quarantine_removed_lines=sum(len(e.removed_lines) for e in quarantines),
        provider_filtered=result.provider_filtered,
        finish_reasons=result.finish_reasons,
        world_state=world,
        check_results=check_results,
        passed=passed,
        failure_reason=reason,
        error=result.error,
    )


def run_suites(
    pipeline: Pipeline,
    cases: list[TestCase],
    run_index: int = 0,
    on_result=None,
) -> SuiteReport:
    """Run every case through one condition.

    `on_result` is called with each RunResult as it completes, so a demo can
    stream one line per case rather than going quiet for several minutes.
    """
    results: list[RunResult] = []
    for case in cases:
        result = run_case(pipeline, case, run_index=run_index)
        results.append(result)
        if on_result is not None:
            on_result(result)

    reset_environment()  # leave the sandbox clean for the next command

    models = [r.backbone_model for r in results if r.backbone_model != "(none)"]
    return SuiteReport(
        condition=pipeline.condition,
        suites=sorted({c.suite for c in cases}),
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(),
        backbone_model=models[0] if models else "(none)",
    )


def load_cases(
    suites: list[str] | None = None,
    limit: int | None = None,
    root: Path | None = None,
) -> list[TestCase]:
    """Load test cases, optionally narrowed to some suites and a count."""
    cases = load_suites(root or settings.TESTSUITES_DIR, suites)
    if limit is not None:
        cases = cases[:limit]
    return cases


def _slug(text: str) -> str:
    """Filesystem-safe form of a model id (they contain / and :)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "unknown"


def write_report(
    report: SuiteReport, path: Path | None = None, phase: str = "phase1"
) -> Path:
    """Persist the run to results/ (gitignored).

    The filename carries the backbone, not just the condition. Earlier runs
    all wrote `phase1_condition_a.json`, so each new backbone silently
    destroyed the previous one's evidence - a cross-model comparison then had
    nothing behind it but numbers quoted from a terminal that had scrolled
    away. Runs now accumulate instead.
    """
    if path is None:
        name = f"{phase}_condition_{report.condition.lower()}_{_slug(report.backbone_model)}.json"
        path = settings.RESULTS_DIR / name
    written = report.write(path)
    logger.info("Wrote %d result(s) to %s", len(report.results), written)
    return written
