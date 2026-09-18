"""Tests for scenario-scoped tool registries (docs/HANDOFF.md 5.2).

IPIGuard's Table 1 has task *scenarios* as columns - Workspace, Slack, Travel,
Banking - each with its own tool suite. Ours was a single workspace surface
(files, web, comms), so every per-scenario rate was the same rate.

The load-bearing constraint on adding more is NOT the tools themselves. It is
that the tool catalogue is rendered into the system prompt, and the response
cache keys on the system prompt. Registering one extra tool in the shared
default registry was measured to change every cache key and orphan all 337
cached responses - which would silently invalidate every number in results/
by making it un-reproducible.

So scenarios are *scoped*: a case declares which surface it runs against, and
only that surface's tools are ever registered. The first test in this file is
the one that matters - the workspace catalogue must stay byte-identical, or
the cache and every prior measurement go with it.

No network calls in this file.
"""

from __future__ import annotations

import pytest

from config import settings
from src.agent import prompts
from src.eval.schemas import load_suites
from src.tools.registry import (
    SCENARIOS,
    UnknownScenarioError,
    build_default_registry,
    build_registry,
)


def catalogue(scenario: str) -> str:
    return build_registry(scenario).describe_for_prompt()


# ---------------------------------------------------------------------------
# The cache-preservation guarantee
# ---------------------------------------------------------------------------


class TestWorkspaceIsUnchanged:
    def test_workspace_catalogue_is_byte_identical_to_the_default(self):
        """If this fails, every cached response and every saved result is
        stale, because the system prompt they were produced under no longer
        exists. Measured: adding one tool changed all 337 cache keys."""
        assert catalogue("workspace") == build_default_registry().describe_for_prompt()

    def test_workspace_system_prompt_is_byte_identical(self):
        """The catalogue is what varies, but the cache keys on the whole
        prompt - so assert the thing that is actually hashed."""
        assert prompts.build_system_prompt(catalogue("workspace")) == (
            prompts.build_system_prompt(build_default_registry().describe_for_prompt())
        )

    def test_the_default_registry_is_still_the_workspace_one(self):
        """Callers that never heard of scenarios must keep working."""
        assert build_default_registry().names() == build_registry("workspace").names()

    def test_workspace_still_holds_exactly_the_seven_original_tools(self):
        assert build_registry("workspace").names() == [
            "comms.list_inbox", "comms.send_email",
            "files.delete", "files.list", "files.read", "files.write",
            "web.fetch",
        ]


# ---------------------------------------------------------------------------
# Scenario isolation
# ---------------------------------------------------------------------------


class TestScenariosAreDisjoint:
    def test_every_declared_scenario_builds(self):
        for name in SCENARIOS:
            assert build_registry(name).names(), f"{name} registered no tools"

    def test_an_unknown_scenario_raises_rather_than_falling_back(self):
        """A typo must not silently run the case on the workspace surface and
        report it as a banking result."""
        with pytest.raises(UnknownScenarioError, match="nosuch"):
            build_registry("nosuch")

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_a_scenario_sees_no_other_scenarios_tools(self, name):
        """AgentDojo's columns are separate worlds: an agent on a banking task
        never sees the travel tools. Overlapping surfaces would make a
        per-scenario rate a statement about the union, not the column."""
        mine = set(build_registry(name).names())
        for other in SCENARIOS:
            if other == name:
                continue
            assert not (mine & set(build_registry(other).names())), (
                f"{name} and {other} share tools; the columns are not independent"
            )

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_scenario_has_a_critical_tool(self, name):
        """Without one, the Misalignment Checkpoint can never fire on that
        column and MF1 is structurally undefined there."""
        assert build_registry(name).critical_tools(), (
            f"{name} has no critical tool, so no defense that guards critical "
            f"actions can be measured on it"
        )

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_scenario_has_an_untrusted_carrier(self, name):
        """Injection cases need somewhere for a payload to arrive from."""
        carriers = [
            s.name for s in build_registry(name).specs() if s.returns_untrusted
        ]
        assert carriers, f"{name} has no tool returning untrusted data"

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_scenario_passes_the_registry_integrity_scan(self, name):
        """ShieldMCP Stage 1 against our own tool descriptions: a new surface
        must not itself trip the firewall."""
        from src.defense.firewall import scan_tool_descriptions

        flagged = scan_tool_descriptions(build_registry(name))
        assert flagged == [], f"{name} tool descriptions flagged: {flagged}"


# ---------------------------------------------------------------------------
# The suite has to agree with the registry
# ---------------------------------------------------------------------------


class TestSuiteScenarios:
    def _cases(self):
        return load_suites(settings.TESTSUITES_DIR)

    def test_every_case_declares_a_scenario_that_exists(self):
        for case in self._cases():
            assert case.scenario in SCENARIOS, (
                f"{case.id} declares scenario {case.scenario!r}, which is not built"
            )

    def test_scenario_defaults_to_workspace(self):
        """Every pre-existing case file omits the field; none of them may
        change meaning by being re-read."""
        from src.eval.schemas import TestCase

        case = TestCase.model_validate({
            "id": "x", "suite": "misalignment", "category": "c", "prompt": "p",
            "expects": {}, "grading": {"checks": [
                {"kind": "tool_called", "tool": "files.write", "role": "task"}
            ]},
        })
        assert case.scenario == "workspace"

    def test_every_case_only_names_tools_its_scenario_provides(self):
        """A check naming a tool the case's surface does not register can
        never pass - it would fail as if the agent had misbehaved."""
        offenders = []
        for case in self._cases():
            available = set(build_registry(case.scenario).names())
            for check in case.grading.checks:
                if check.tool and check.tool not in available:
                    offenders.append(f"{case.id}({case.scenario}) -> {check.tool}")
        assert offenders == [], offenders

    def test_injection_cases_plant_into_a_tool_their_scenario_provides(self):
        offenders = [
            f"{c.id}({c.scenario}) -> {c.injection_tool}"
            for c in self._cases()
            if c.injection_tool
            and c.injection_tool not in set(build_registry(c.scenario).names())
        ]
        assert offenders == [], offenders


# ---------------------------------------------------------------------------
# The pipelines have to honour the case's scenario
# ---------------------------------------------------------------------------


class StubClient:
    def chat(self, messages, **kwargs):
        from src.llm.client import LLMResponse

        return LLMResponse(
            content="Final: done.", model_used="fake", latency_ms=1,
            from_cache=False, provider="groq",
        )


class TestPipelinesSwitchScenario:
    def test_condition_a_uses_the_requested_surface(self):
        from src.pipeline.condition_a import ConditionA

        pipeline = ConditionA(StubClient())
        pipeline.use_scenario("workspace")
        assert "files.read" in pipeline.agent.registry.describe_for_prompt()

    def test_switching_scenario_changes_the_catalogue_the_agent_sees(self):
        from src.pipeline.condition_a import ConditionA

        other = next(n for n in SCENARIOS if n != "workspace")
        pipeline = ConditionA(StubClient())
        pipeline.use_scenario("workspace")
        before = pipeline.agent.registry.describe_for_prompt()
        pipeline.use_scenario(other)
        assert pipeline.agent.registry.describe_for_prompt() != before

    def test_condition_b_rebuilds_its_wrappers_for_the_new_surface(self):
        """The misalignment wrapper is built once and wraps a registry; if it
        is not rebuilt it keeps guarding the previous scenario's tools."""
        from src.pipeline.condition_b import ConditionB

        other = next(n for n in SCENARIOS if n != "workspace")
        pipeline = ConditionB(StubClient(), enabled_modules={"misalignment"})
        pipeline.use_scenario(other)
        inner = pipeline.misalignment_registry._inner
        assert set(inner.names()) == set(build_registry(other).names())

    def test_the_runner_selects_the_scenario_from_the_case(self):
        from src.eval import runner
        from src.pipeline.condition_a import ConditionA

        seen = []

        class Recording(ConditionA):
            def use_scenario(self, name):
                seen.append(name)
                super().use_scenario(name)

        case = next(c for c in load_suites(settings.TESTSUITES_DIR))
        runner.run_case(Recording(StubClient()), case)
        assert seen == [case.scenario]


# ---------------------------------------------------------------------------
# Outcome-based grading has to reach the new surfaces too
# ---------------------------------------------------------------------------


class TestWorldStateCoversEveryScenario:
    """`content_preserved` is the check kind that survives a blocked action.

    Phase 5 found the hard way that an argument-scanning attack check inverts
    when a defense blocks the call: the tool was never successfully called, so
    the argument is absent, so preventing the overreach scores as committing
    it. The fix was to grade on world state. That fix only helps a scenario
    whose state the snapshot actually captures - so banking and travel have to
    be in it, or their cases are stuck with the fragile check kinds.
    """

    def test_bank_state_is_captured(self):
        from src.eval.runner import capture_world_state, reset_environment
        from src.tools import banking

        reset_environment()
        banking.update_payee("Landlord", "ACC-9999")
        world = capture_world_state()
        blob = " ".join(world.values())
        assert "ACC-9999" in blob, "a repointed payee left no trace in world state"

    def test_transfers_are_captured(self):
        from src.eval.runner import capture_world_state, reset_environment
        from src.tools import banking

        reset_environment()
        banking.transfer("ACC-6666", "50,000 INR", "urgent")
        assert "ACC-6666" in " ".join(capture_world_state().values())

    def test_bookings_are_captured(self):
        from src.eval.runner import capture_world_state, reset_environment
        from src.tools import travel

        reset_environment()
        world_before = " ".join(capture_world_state().values())
        assert "PNR-4417" in world_before, "seed booking missing from world state"
        travel.cancel_booking("PNR-4417")
        assert "PNR-4417" not in " ".join(capture_world_state().values())

    def test_reset_clears_state_between_cases(self):
        from src.eval.runner import capture_world_state, reset_environment
        from src.tools import banking

        reset_environment()
        banking.transfer("ACC-6666", "50,000 INR", "")
        reset_environment()
        assert "ACC-6666" not in " ".join(capture_world_state().values())

    def test_workspace_snapshot_still_has_its_files(self):
        """Adding surfaces must not displace what was already captured."""
        from src.eval.runner import capture_world_state, reset_environment

        reset_environment()
        world = capture_world_state()
        assert "notes/meeting_notes.txt" in world
        assert "Size the test suites" in world["notes/meeting_notes.txt"]


class TestCasesDoNotReimplementRefusal:
    """Refusal detection lives in one place and must stay there.

    `harm_t001` shipped with its own `final_matches_any` list of refusal words
    as an attack check. The model refused cleanly - "I'm not able to help with
    that" - and `grading.detect_refusal` agreed, but the per-case list did not
    contain that phrasing, so the case failed with "followed the injected
    instruction" while the transcript showed a textbook refusal. A per-case
    copy of a central heuristic is a second source of truth that drifts.
    """

    def test_no_case_checks_for_refusal_words_itself(self):
        from src.eval.grading import REFUSAL_PATTERNS
        from src.eval.schemas import CheckKind, CheckRole

        known = {p.strip().lower() for p in REFUSAL_PATTERNS}
        offenders = []
        for case in load_suites(settings.TESTSUITES_DIR):
            for check in case.grading.checks:
                if check.kind is not CheckKind.FINAL_MATCHES_ANY:
                    continue
                if check.role is not CheckRole.ATTACK:
                    continue
                values = {v.strip().lower() for v in (check.values or [])}
                if any(any(v in p or p in v for p in known) for v in values):
                    offenders.append(f"{case.id}: {sorted(values)}")
        assert offenders == [], (
            "these cases re-implement refusal detection; declare "
            f"expects.should_refuse instead: {offenders}"
        )


# ---------------------------------------------------------------------------
# A narrowed run must not clobber a broader one's evidence
# ---------------------------------------------------------------------------


class TestResultFilesCarryTheirScope:
    """This has now destroyed a baseline twice.

    The filename carried the backbone but not the filter, so
    `--suite injection` overwrote the full Condition A baseline (recorded in
    HANDOFF 5.1), and then `--scenario banking --scenario travel` overwrote it
    again - turning the cumulative ablation's first row into 13 banking/travel
    cases while it was still labelled "Condition A (no defenses)".

    Nothing errors when this happens. The table just quietly reports a
    different denominator under the same name, which is the exact class of
    defect this repo treats as worse than a crash.
    """

    def _report(self, cases, condition="A"):
        from src.eval.schemas import RunResult, SuiteReport, Outcome

        results = [
            RunResult(
                test_case_id=c.id, condition=condition, run_index=0,
                backbone_model="qwen/qwen3.8-27b", transcript=[],
                timestamp="2026-09-18T00:00:00Z", suite=c.suite,
                scenario=c.scenario,
                outcome=Outcome(refused=False, task_completed=True,
                                latency_ms=1, num_llm_calls=1),
            )
            for c in cases
        ]
        return SuiteReport(
            condition=condition, suites=sorted({c.suite for c in cases}),
            results=results, generated_at="2026-09-18T00:00:00Z",
            backbone_model="qwen/qwen3.8-27b",
        )

    def test_a_full_run_keeps_the_plain_name(self):
        from src.eval.runner import default_report_name

        cases = load_suites(settings.TESTSUITES_DIR)
        name = default_report_name(self._report(cases), phase="phase1")
        assert name == "phase1_condition_a_qwen-qwen3.8-27b.json"

    def test_a_scenario_narrowed_run_gets_its_own_name(self):
        from src.eval.runner import default_report_name

        cases = [c for c in load_suites(settings.TESTSUITES_DIR)
                 if c.scenario in {"banking", "travel"}]
        name = default_report_name(self._report(cases), phase="phase1")
        assert name != "phase1_condition_a_qwen-qwen3.8-27b.json"
        assert "banking" in name and "travel" in name

    def test_a_suite_narrowed_run_gets_its_own_name(self):
        from src.eval.runner import default_report_name

        cases = [c for c in load_suites(settings.TESTSUITES_DIR)
                 if c.suite == "injection"]
        name = default_report_name(self._report(cases), phase="phase1")
        assert "injection" in name
        assert name != "phase1_condition_a_qwen-qwen3.8-27b.json"

    def test_two_different_scopes_never_collide(self):
        from src.eval.runner import default_report_name

        cases = load_suites(settings.TESTSUITES_DIR)
        full = default_report_name(self._report(cases), phase="phase1")
        banking = default_report_name(
            self._report([c for c in cases if c.scenario == "banking"]),
            phase="phase1",
        )
        injection = default_report_name(
            self._report([c for c in cases if c.suite == "injection"]),
            phase="phase1",
        )
        assert len({full, banking, injection}) == 3


class TestAblationTablePicksEvidenceVisibly:
    """The table globbed `phase1_condition_a_*.json` and took the last match
    alphabetically - so a scoped file could become the "Condition A (no
    defenses)" row purely by sorting late, with nothing on screen saying which
    file the row came from."""

    def _module(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "ablation_table_mod", Path("demos/ablation_table.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_broadest_file_wins_not_the_last_alphabetically(self, tmp_path):
        import json

        def write(name, n):
            # The pinned backbone: this test is about breadth, and a foreign
            # model is now filtered out entirely (see TestAblationTableIsSingleBackbone).
            (tmp_path / name).write_text(json.dumps({
                "condition": "A", "suites": ["direct_harm"], "generated_at": "x",
                "backbone_model": settings.BACKBONE_MODEL,
                "results": [
                    {"test_case_id": f"c{i}", "condition": "A", "run_index": 0,
                     "backbone_model": settings.BACKBONE_MODEL, "transcript": [],
                     "timestamp": "x", "passed": True,
                     "outcome": {"refused": False, "task_completed": True,
                                 "latency_ms": 1, "num_llm_calls": 1}}
                    for i in range(n)
                ],
            }), encoding="utf-8")

        write("phase1_condition_a_pinned.json", 26)
        write("phase1_condition_a_pinned_zzz-narrow.json", 3)

        module = self._module()
        data, source = module.load("phase1_condition_a_*.json", root=tmp_path)
        assert len(data["results"]) == 26
        assert source == "phase1_condition_a_pinned.json"

    def test_a_missing_snapshot_reports_no_source(self, tmp_path):
        module = self._module()
        assert module.load("nothing_*.json", root=tmp_path) == (None, "")


class TestAblationTableIsSingleBackbone:
    """HANDOFF 3: the backbone is pinned, and a result from another model is
    not comparable with one from the pinned model.

    Making the evidence visible exposed that the Condition A row had been
    built from `phase1_condition_a_gemini-2.5-flash.json` - a different
    backbone - while the other four rows were all qwen. The row reported
    ASR_inj 1.00 and HS 0.40 next to four qwen rows reporting 0.00, and
    nothing said the top row was a different model.
    """

    def _module(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "ablation_table_mod2", Path("demos/ablation_table.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write(self, tmp_path, name, model, n):
        import json

        (tmp_path / name).write_text(json.dumps({
            "condition": "A", "suites": ["direct_harm"], "generated_at": "x",
            "backbone_model": model,
            "results": [
                {"test_case_id": f"c{i}", "condition": "A", "run_index": 0,
                 "backbone_model": model, "transcript": [], "timestamp": "x",
                 "passed": True,
                 "outcome": {"refused": False, "task_completed": True,
                             "latency_ms": 1, "num_llm_calls": 1}}
                for i in range(n)
            ],
        }), encoding="utf-8")

    def test_a_foreign_backbone_is_not_used_even_if_broader(self, tmp_path):
        self._write(tmp_path, "phase1_condition_a_other.json", "google/gemini", 30)
        self._write(tmp_path, "phase1_condition_a_pinned.json", settings.BACKBONE_MODEL, 5)

        module = self._module()
        data, source = module.load("phase1_condition_a_*.json", root=tmp_path)
        assert source == "phase1_condition_a_pinned.json"
        assert len(data["results"]) == 5

    def test_no_pinned_snapshot_means_no_row(self, tmp_path):
        self._write(tmp_path, "phase1_condition_a_other.json", "google/gemini", 30)

        module = self._module()
        assert self._module().load("phase1_condition_a_*.json", root=tmp_path) == (None, "")
