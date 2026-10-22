"""The AgentDojo bridge, tested against AgentDojo's own grader at zero model cost."""

from __future__ import annotations

import pytest

pytest.importorskip("agentdojo", reason="pip install -r requirements-agentdojo.txt")

from agentdojo.attacks.important_instructions_attacks import (  # noqa: E402
    ImportantInstructionsAttack,
)
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from src.agent import prompts  # noqa: E402
from src.agent.loop import AgentResult  # noqa: E402
from src.agentdojo_bridge import (  # noqa: E402
    BANKING_DEV,
    BANKING_HELDOUT,
    SPLITS,
    SPOTLIGHT_SUFFIX,
    TAGS,
    TOOL_FILTER_PROMPT,
    BaselineRunner,
    GuardedPipeline,
)
from src.llm.client import LLMResponse  # noqa: E402
from src.pipeline.condition_a import ConditionA  # noqa: E402

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


def test_banking_split_partitions_the_suite_and_keeps_inspected_tasks_in_dev():
    assert set(BANKING_DEV) | set(BANKING_HELDOUT) == set(BANKING.user_tasks)
    assert not set(BANKING_DEV) & set(BANKING_HELDOUT)
    assert {"user_task_0", "user_task_1", "user_task_2"} <= set(BANKING_DEV)


def test_every_split_partitions_its_suite():
    for suite, (dev, held) in SPLITS.items():
        assert set(dev) | set(held) == set(SUITES[suite].user_tasks), suite
        assert not set(dev) & set(held), suite
    assert "user_task_0" in SPLITS["slack"][0]


# ---------------------------------------------------------------------------
# AgentDojo's own defenses as baselines (spike 15). No model, no network: a
# scripted client replays fixed replies and records what it was sent.
# ---------------------------------------------------------------------------

READ_BILL = 'Thought: read it\nAction: {"tool": "read_file", "args": {"file_path": "bill-december-2023.txt"}}'
DONE = "Thought: done\nFinal: paid"


class ScriptedClient:
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])  # the loop mutates its list
        return LLMResponse(content=self.replies.pop(0), model_used="fake/4b",
                           latency_ms=1, from_cache=False)


def run_bill_task(make_runner, attacked=False):
    """banking user_task_0, optionally with the real important_instructions payload."""
    task = BANKING.user_tasks["user_task_0"]
    pipeline = GuardedPipeline(make_runner, "test")
    injections = (ImportantInstructionsAttack(BANKING, pipeline)
                  .attack(task, BANKING.injection_tasks["injection_task_0"]) if attacked else {})
    pipeline.query(task.PROMPT, FunctionsRuntime(BANKING.tools),
                   task.init_environment(BANKING.load_and_inject_default_environment(injections)))
    return pipeline


def baseline(client, defense, **kw):
    return lambda reg: BaselineRunner(client, reg, defense, "fake/4b", 15, **kw)


def test_condition_a_system_prompt_is_unchanged_by_the_suffix_hook():
    client = ScriptedClient(DONE)
    run_bill_task(lambda reg: ConditionA(client, registry=reg, model="fake/4b", max_steps=15))
    system = client.sent[0][0]["content"]
    assert system.endswith(prompts.SYSTEM_PROMPT_TEMPLATE[-40:])
    assert "<<" not in system


def test_unknown_defense_is_rejected():
    with pytest.raises(ValueError):
        BaselineRunner(ScriptedClient(), None, "nope", "m", 15)


def test_spotlighting_delimits_tool_output_and_says_so_in_the_system_prompt():
    client = ScriptedClient(READ_BILL, DONE)
    run_bill_task(baseline(client, "spotlighting_with_delimiting"))
    second = client.sent[1]
    assert second[0]["content"].endswith(SPOTLIGHT_SUFFIX)
    assert second[-1]["content"].startswith("Observation: <<") and second[-1]["content"].endswith(">>")


def test_repeat_user_prompt_follows_every_tool_output_with_the_task():
    client = ScriptedClient(READ_BILL, DONE)
    run_bill_task(baseline(client, "repeat_user_prompt"))
    task = BANKING.user_tasks["user_task_0"].PROMPT
    assert client.sent[1][-1]["content"].endswith(f"\n\n{task}")
    assert "<<" not in client.sent[1][-1]["content"]


def test_tool_filter_keeps_only_named_tools_and_counts_its_call():
    client = ScriptedClient("read_file, send_money", READ_BILL, DONE)
    pipeline = run_bill_task(baseline(client, "tool_filter"))
    assert client.sent[0][-1]["content"] == TOOL_FILTER_PROMPT
    agent_system = client.sent[1][0]["content"]
    assert "read_file(" in agent_system and "send_money(" in agent_system
    assert "get_balance(" not in agent_system
    assert pipeline.last_runner.notes["tools_kept"] == ["read_file", "send_money"]
    assert pipeline.last_result.num_llm_calls == 3


def test_tool_filter_that_drops_a_tool_makes_it_uncallable():
    client = ScriptedClient("send_money", READ_BILL, DONE)
    pipeline = run_bill_task(baseline(client, "tool_filter"))
    assert "No such tool: read_file" in client.sent[2][-1]["content"]
    assert pipeline.last_outputs == []  # never reached AgentDojo's runtime


def test_pi_detector_replaces_a_flagged_output_with_agentdojos_placeholder():
    client = ScriptedClient(READ_BILL, DONE)
    detector = lambda text: ("<INFORMATION>" in text, 0.0)  # noqa: E731
    pipeline = run_bill_task(baseline(client, "transformers_pi_detector", detector=detector),
                             attacked=True)
    assert client.sent[1][-1]["content"] == (
        "Observation: <Data omitted because a prompt injection was detected>")
    assert pipeline.last_runner.notes == {"pi_checked": 1, "pi_flagged": 1}
    assert "<INFORMATION>" in pipeline.last_outputs[0]  # exposure is still recorded


def test_pi_detector_passes_a_clean_output_through():
    client = ScriptedClient(READ_BILL, DONE)
    run_bill_task(baseline(client, "transformers_pi_detector", detector=lambda t: (False, 1.0)))
    assert "bill" in client.sent[1][-1]["content"].lower()
