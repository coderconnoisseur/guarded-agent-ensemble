"""The AgentDojo bridge, tested against AgentDojo's own grader at zero model cost."""

from __future__ import annotations

import pytest

pytest.importorskip("agentdojo", reason="pip install -r requirements-agentdojo.txt")

from agentdojo.attacks.important_instructions_attacks import (  # noqa: E402
    ImportantInstructionsAttack,
)
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from src.agent.loop import AgentResult  # noqa: E402
from src.agentdojo_bridge import TAGS, GuardedPipeline  # noqa: E402

SUITES = get_suites("v1.2.2")
BANKING = SUITES["banking"]


class Scripted:
    """Stands in for ConditionA: dispatches fixed calls through the registry."""

    def __init__(self, registry, calls, answer):
        self.registry, self.calls, self.answer = registry, calls, answer

    def run(self, task):
        for name, args in self.calls:
            self.registry.dispatch(name, args)
        return AgentResult(task=task, final_answer=self.answer, stop_reason="final")


def ground_truth_pipeline(suite, user_task):
    env = user_task.init_environment(suite.load_and_inject_default_environment({}))
    calls = [(c.function, dict(c.args)) for c in user_task.ground_truth(env)]
    return GuardedPipeline(
        lambda reg: Scripted(reg, calls, user_task.GROUND_TRUTH_OUTPUT), "test"
    )


def test_every_tool_is_tagged_exactly_once():
    names = {f.name for s in SUITES.values() for f in s.tools}
    assert names == set(TAGS)


def test_critical_tools_are_never_read_only():
    assert not [n for n, (crit, ro, _) in TAGS.items() if crit and ro]


@pytest.mark.parametrize("task_id", sorted(BANKING.user_tasks))
def test_ground_truth_through_our_registry_is_scored_as_solved(task_id):
    """The whole adapter, end to end: our registry dispatches, AgentDojo's
    runtime mutates its env, AgentDojo's grader reads env + trace + answer."""
    task = BANKING.user_tasks[task_id]
    utility, _ = BANKING.run_task_with_pipeline(ground_truth_pipeline(BANKING, task), task, None, {})
    assert utility, f"{task_id}: ground truth via the bridge was not scored as solved"


def test_a_call_that_never_reaches_the_registry_is_not_in_the_trace():
    """A defense-blocked call must not be graded as an executed one."""
    task = BANKING.user_tasks["user_task_0"]
    pipeline = GuardedPipeline(lambda reg: Scripted(reg, [], "done"), "test")
    _, _, env, messages, _ = pipeline.query(
        task.PROMPT, FunctionsRuntime(BANKING.tools),
        task.init_environment(BANKING.load_and_inject_default_environment({})),
    )
    assert not any(m["role"] == "assistant" and m["tool_calls"] for m in messages)
    assert messages[-1]["role"] == "assistant"


def test_failed_tool_call_comes_back_as_an_error_observation():
    task = BANKING.user_tasks["user_task_0"]
    seen = {}

    class Bad(Scripted):
        def run(self, task):
            seen["r"] = self.registry.dispatch("send_money", {"recipient": "x", "amount": "not a number",
                                                              "subject": "s", "date": "2022-01-01"})
            return AgentResult(task=task, final_answer="", stop_reason="final")

    GuardedPipeline(lambda reg: Bad(reg, [], ""), "test").query(
        task.PROMPT, FunctionsRuntime(BANKING.tools),
        task.init_environment(BANKING.load_and_inject_default_environment({})),
    )
    assert not seen["r"].ok and "ValidationError" in seen["r"].error


def test_the_attack_accepts_the_pipeline_name():
    pipeline = GuardedPipeline(lambda reg: None, "condition-a")
    attack = ImportantInstructionsAttack(BANKING, pipeline)
    task = BANKING.user_tasks["user_task_0"]
    injections = attack.attack(task, BANKING.injection_tasks["injection_task_0"])
    assert any("Local model" in p for p in injections.values())
