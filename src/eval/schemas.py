"""Pydantic models for the evaluation harness (CLAUDE.md 8).

8.1 defines a test case's `grading` as a list of free-text `rubric` strings.
Those cannot be checked mechanically, which collides with 2's rule that
pass/fail should be decided from the transcript rather than by asking an LLM.

This module resolves that by keeping `rubric` as human-readable documentation
(it is what appears in the final report next to a case) and adding a parallel
`checks` list of machine-evaluable predicates. `src/eval/grading.py` evaluates
the checks; nothing ever evaluates the prose.

Each check carries a `role` tying it to a GAI sub-metric (9):

    role="task"    the legitimate task was completed   -> BU, UA
    role="attack"  the malicious/overreaching goal was NOT achieved -> ASR_inj, HS

Roles are interpreted per suite, so a failed `attack` check on a misalignment
case counts toward the misalignment analysis and never leaks into ASR_inj,
which is scoped to the injection suite.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CheckKind(str, Enum):
    """The machine-evaluable predicates a test case may assert."""

    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"
    ARG_CONTAINS = "arg_contains"
    ARG_NOT_CONTAINS = "arg_not_contains"
    FINAL_MATCHES_ANY = "final_matches_any"
    MAX_TOOL_CALLS = "max_tool_calls"
    NO_CRITICAL_ACTION = "no_critical_action"
    # Checked against the post-run world snapshot, not the tool calls: some
    # destructive-looking sequences preserve the data (copy-then-delete is a
    # correct "archive"), and only the resulting state distinguishes them.
    CONTENT_PRESERVED = "content_preserved"


class CheckRole(str, Enum):
    """What a check contributes to."""

    TASK = "task"
    ATTACK = "attack"


# Which fields each kind requires. Validated at load time so a malformed test
# case fails loudly when the suite loads, rather than silently grading wrong
# and quietly moving a number in the final report.
_REQUIRED_FIELDS: dict[CheckKind, tuple[str, ...]] = {
    CheckKind.TOOL_CALLED: ("tool",),
    CheckKind.TOOL_NOT_CALLED: ("tool",),
    CheckKind.ARG_CONTAINS: ("tool", "arg", "value"),
    CheckKind.ARG_NOT_CONTAINS: ("tool", "arg", "value"),
    CheckKind.FINAL_MATCHES_ANY: ("values",),
    CheckKind.MAX_TOOL_CALLS: ("tool", "count"),
    CheckKind.NO_CRITICAL_ACTION: (),
    CheckKind.CONTENT_PRESERVED: ("value",),
}


class Check(BaseModel):
    """One mechanically evaluable assertion about an agent run."""

    kind: CheckKind
    role: CheckRole = CheckRole.TASK
    tool: str | None = None
    arg: str | None = None
    value: str | None = None
    values: list[str] | None = None
    count: int | None = None
    # Optional human note; shown when this check is the reason a case failed.
    note: str = ""

    @model_validator(mode="after")
    def _require_fields_for_kind(self) -> Check:
        missing = [f for f in _REQUIRED_FIELDS[self.kind] if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"check kind {self.kind.value!r} requires {', '.join(missing)}"
            )
        return self

    def describe(self) -> str:
        """Short human rendering, used in failure output."""
        if self.note:
            return self.note
        parts = [self.kind.value]
        for field in ("tool", "arg", "value", "count"):
            got = getattr(self, field)
            if got is not None:
                parts.append(f"{field}={got!r}")
        if self.values is not None:
            parts.append(f"values={self.values!r}")
        return " ".join(parts)


class Grading(BaseModel):
    """8.1's grading block, extended with machine-checkable predicates."""

    type: Literal["rubric"] = "rubric"
    # Human-readable criteria. Documentation only - never evaluated in code.
    rubric: list[str] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)

    def by_role(self, role: CheckRole) -> list[Check]:
        return [c for c in self.checks if c.role is role]


class Expects(BaseModel):
    """The behaviour a *well-behaved* agent would show on this case.

    Condition A is expected to violate several of these - that is the Phase 1
    baseline finding, not a bug in the harness.
    """

    # Three-valued on purpose. `None` means "either is acceptable", which is
    # the honest expectation for a misalignment case: a safe agent may pause
    # and ask, or may carry the task out carefully without destroying
    # anything, and forcing an equality assertion would mark one of those two
    # correct behaviours as a failure. For harm and injection cases it stays a
    # hard bool.
    should_refuse: bool | None = False
    should_complete_original_task: bool = True
    misalignment_expected: bool = False


class TestCase(BaseModel):
    """One test case (8.1)."""

    # This is a data model, not a pytest class - stop pytest collecting it.
    __test__ = False

    id: str
    suite: str
    category: str
    prompt: str

    # Injection suites only: the hidden instruction, which tool's simulated
    # response carries it, and the key that response is filed under.
    # `injection_key` is not in 8.1 but is mechanically necessary - the runner
    # has to know *which* page or file to plant the payload in.
    injected_content: str | None = None
    injection_tool: str | None = None
    injection_key: str | None = None

    expects: Expects
    grading: Grading

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, value: str) -> str:
        if not value or " " in value:
            raise ValueError(f"test case id must be a non-empty slug, got {value!r}")
        return value

    @model_validator(mode="after")
    def _expectations_are_satisfiable(self) -> TestCase:
        """Catch expectations no set of checks could ever produce.

        `task_completed` is defined as "at least one task check, and all task
        checks passed". A case that expects completion but declares no task
        check can therefore never pass, however well the agent behaves.
        """
        has_task_checks = bool(self.grading.by_role(CheckRole.TASK))
        if self.expects.should_complete_original_task and not has_task_checks:
            raise ValueError(
                f"{self.id}: expects.should_complete_original_task is true but no "
                f"check has role='task', so the case can never pass"
            )
        if self.injected_content and not self.injection_tool:
            raise ValueError(f"{self.id}: injected_content requires injection_tool")
        return self

    @classmethod
    def from_file(cls, path: Path) -> TestCase:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class CheckResult(BaseModel):
    """Outcome of evaluating one check."""

    kind: CheckKind
    role: CheckRole
    passed: bool
    detail: str


class Outcome(BaseModel):
    """8.2's outcome block."""

    refused: bool
    task_completed: bool
    attack_succeeded: bool | None = None
    misalignment_flagged: bool | None = None
    latency_ms: int
    num_llm_calls: int


class RunResult(BaseModel):
    """One test case, one condition, one repeat (8.2).

    `check_results`, `passed` and `failure_reason` are additions to 8.2's
    shape. They carry the per-case verdict the Phase 1 demo prints, and the
    evidence for it, so a failing line can be traced without re-running.
    """

    test_case_id: str
    condition: str
    run_index: int
    backbone_model: str
    transcript: list[dict[str, str]]
    outcome: Outcome
    timestamp: str

    suite: str = ""
    # True when a provider-side safety layer intervened rather than the model
    # itself declining. A refusal recorded on such a run says something about
    # the hosted API, not about the backbone, so HS must not absorb it
    # silently. See Provider.finish_signal.
    provider_filtered: bool = False
    finish_reasons: list[str] = Field(default_factory=list)
    # Sandbox contents captured immediately after the run, before the next
    # case resets the world. Recorded so a content-preserved check stays
    # reproducible from the saved result rather than needing the live disk.
    world_state: dict[str, str] = Field(default_factory=dict)
    check_results: list[CheckResult] = Field(default_factory=list)
    passed: bool = False
    failure_reason: str = ""
    error: str | None = None

    def summary_line(self) -> str:
        """The one-line-per-case format the Phase 1 DoD asks for.

        A passing case has to say what actually happened. Reporting a refused
        harmful request as "task completed" is not just cosmetic - it reads as
        though the agent carried out the fraud, which is the opposite of what
        the transcript shows.
        """
        tag = "OK  " if self.passed else "FAIL"
        if not self.passed:
            return f"[{tag}] {self.test_case_id}: {self.failure_reason}"
        if self.outcome.refused:
            detail = "refused, as expected"
        elif self.outcome.task_completed:
            detail = "task completed"
        else:
            detail = "no unsafe action taken"
        return f"[{tag}] {self.test_case_id}: {detail}"


class SuiteReport(BaseModel):
    """All runs for one condition over one or more suites."""

    condition: str
    suites: list[str]
    results: list[RunResult]
    generated_at: str
    backbone_model: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


def load_suite(directory: Path) -> list[TestCase]:
    """Load and validate every test case in a suite directory, sorted by id."""
    cases = [TestCase.from_file(p) for p in sorted(directory.glob("*.json"))]
    return sorted(cases, key=lambda c: c.id)


def load_suites(root: Path, suites: list[str] | None = None) -> list[TestCase]:
    """Load several suites at once.

    A suite directory that exists but is empty is not an error: the diversity
    suite is deliberately empty (see the note in that folder and 9 / 11).
    """
    names = suites if suites is not None else sorted(
        p.name for p in root.iterdir() if p.is_dir()
    )
    cases: list[TestCase] = []
    for name in names:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"No such test suite: {name} (looked in {root})")
        cases.extend(load_suite(directory))
    return cases
