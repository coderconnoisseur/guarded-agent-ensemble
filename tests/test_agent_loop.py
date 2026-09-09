"""Unit tests for the 5.3 prompted-JSON protocol parser and the ReAct loop.

The parser is the project's single point of failure for tool calling: if it
mis-reads a reply, Condition A looks worse than it is and every downstream
number moves. CLAUDE.md 12 asks for tests on exactly this.
"""

from __future__ import annotations

import pytest

from src.agent.loop import AgentResult, ReActAgent, parse_agent_reply
from src.llm.client import LLMResponse
from src.tools.registry import ToolParam, ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_canonical_shape(self):
        reply = 'Thought: I should read it.\nAction: {"tool": "files.read", "args": {"path": "a.txt"}}'
        parsed = parse_agent_reply(reply)
        assert parsed.kind == "action"
        assert parsed.tool == "files.read"
        assert parsed.args == {"path": "a.txt"}
        assert parsed.thought == "I should read it."

    def test_empty_args_object(self):
        parsed = parse_agent_reply('Action: {"tool": "comms.list_inbox", "args": {}}')
        assert parsed.kind == "action"
        assert parsed.args == {}

    def test_missing_args_key_defaults_to_empty(self):
        parsed = parse_agent_reply('Action: {"tool": "comms.list_inbox"}')
        assert parsed.kind == "action"
        assert parsed.args == {}

    def test_tolerates_code_fences(self):
        reply = 'Thought: ok\nAction:\n```json\n{"tool": "files.list", "args": {}}\n```'
        assert parse_agent_reply(reply).tool == "files.list"

    def test_tolerates_markdown_bold_markers(self):
        reply = '**Thought:** ok\n**Action:** {"tool": "files.list", "args": {}}'
        assert parse_agent_reply(reply).tool == "files.list"

    def test_accepts_name_and_arguments_aliases(self):
        parsed = parse_agent_reply('Action: {"name": "files.read", "arguments": {"path": "x"}}')
        assert parsed.tool == "files.read"
        assert parsed.args == {"path": "x"}

    def test_double_encoded_args_string_is_decoded(self):
        reply = 'Action: {"tool": "files.read", "args": "{\\"path\\": \\"x.txt\\"}"}'
        parsed = parse_agent_reply(reply)
        assert parsed.kind == "action"
        assert parsed.args == {"path": "x.txt"}

    def test_braces_inside_a_string_value_do_not_truncate_the_payload(self):
        """A naive regex would cut this off at the first inner brace."""
        reply = (
            'Action: {"tool": "comms.send_email", "args": '
            '{"to": "a@b.com", "subject": "hi", "body": "use {curly} braces"}}'
        )
        parsed = parse_agent_reply(reply)
        assert parsed.kind == "action"
        assert parsed.args["body"] == "use {curly} braces"

    def test_trailing_prose_after_the_json_is_ignored(self):
        reply = 'Action: {"tool": "files.list", "args": {}}\nI will check the result next.'
        assert parse_agent_reply(reply).kind == "action"


class TestParseFinal:
    def test_canonical_shape(self):
        parsed = parse_agent_reply("Thought: Done.\nFinal: The file welcomes the user.")
        assert parsed.kind == "final"
        assert parsed.final_text == "The file welcomes the user."
        assert parsed.thought == "Done."

    def test_final_answer_spelling(self):
        assert parse_agent_reply("Final Answer: 42").kind == "final"

    def test_multiline_answer_is_kept_whole(self):
        parsed = parse_agent_reply("Final: line one\nline two")
        assert parsed.final_text == "line one\nline two"

    def test_empty_final_is_a_parse_failure(self):
        assert parse_agent_reply("Final:").kind == "unparseable"


class TestParseFailures:
    def test_empty_reply(self):
        assert parse_agent_reply("").kind == "unparseable"

    def test_no_marker_at_all(self):
        parsed = parse_agent_reply("Sure, I can help with that!")
        assert parsed.kind == "unparseable"
        assert "Action" in parsed.problem

    def test_malformed_json(self):
        parsed = parse_agent_reply('Action: {"tool": "files.read", args: broken}')
        assert parsed.kind == "unparseable"
        assert "malformed" in parsed.problem

    def test_unclosed_json_object(self):
        parsed = parse_agent_reply('Action: {"tool": "files.read", "args": {"path": "a"}')
        assert parsed.kind == "unparseable"
        assert "not closed" in parsed.problem

    def test_missing_tool_key(self):
        parsed = parse_agent_reply('Action: {"args": {"path": "a"}}')
        assert parsed.kind == "unparseable"
        assert "tool" in parsed.problem

    def test_args_of_the_wrong_type(self):
        parsed = parse_agent_reply('Action: {"tool": "files.read", "args": [1, 2]}')
        assert parsed.kind == "unparseable"

    def test_both_markers_takes_the_earlier_one(self):
        reply = 'Action: {"tool": "files.list", "args": {}}\nFinal: all done'
        assert parse_agent_reply(reply).kind == "action"
        reply2 = 'Final: all done\nAction: {"tool": "files.list", "args": {}}'
        assert parse_agent_reply(reply2).kind == "final"


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class ScriptedClient:
    """Stands in for LLMClient, replaying a fixed list of replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        content = self.replies.pop(0) if self.replies else "Final: out of script"
        return LLMResponse(
            content=content, model_used="test/model", latency_ms=1,
            from_cache=False, raw={},
        )


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo.say",
            description="Echo a message back.",
            params=[ToolParam("text", "string", "What to echo.")],
            handler=lambda text: f"echoed: {text}",
            read_only=True,
        )
    )
    reg.register(
        ToolSpec(
            name="danger.wipe",
            description="Delete everything.",
            params=[],
            handler=lambda: "wiped",
            read_only=False,
            critical=True,
        )
    )
    return reg


class TestLoop:
    def test_single_tool_call_then_final(self, registry):
        client = ScriptedClient([
            'Thought: echo it\nAction: {"tool": "echo.say", "args": {"text": "hi"}}',
            "Thought: done\nFinal: I echoed hi.",
        ])
        result = ReActAgent(client, registry).run("echo hi")

        assert result.stop_reason == "final"
        assert result.final_answer == "I echoed hi."
        assert result.num_llm_calls == 2
        assert result.tool_calls == [("echo.say", {"text": "hi"})]
        assert "echoed: hi" in [t["content"] for t in result.transcript]

    def test_transcript_roles_match_the_schema(self, registry):
        client = ScriptedClient([
            'Action: {"tool": "echo.say", "args": {"text": "hi"}}',
            "Final: done",
        ])
        result = ReActAgent(client, registry).run("echo hi")
        roles = [entry["role"] for entry in result.transcript]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert "assistant" in roles and "tool" in roles

    def test_malformed_reply_gets_a_repair_turn(self, registry):
        client = ScriptedClient(["I'll just chat instead.", "Final: recovered"])
        result = ReActAgent(client, registry).run("do a thing")

        assert result.stop_reason == "final"
        assert result.final_answer == "recovered"
        repair = client.calls[-1][-1]["content"]
        assert "could not be parsed" in repair

    def test_gives_up_after_repeated_parse_failures(self, registry):
        client = ScriptedClient(["nope"] * 10)
        result = ReActAgent(client, registry, max_parse_retries=2).run("x")
        assert result.stop_reason == "parse_failure"
        assert result.num_llm_calls == 3  # first attempt plus two repairs

    def test_unknown_tool_is_an_observation_not_a_crash(self, registry):
        client = ScriptedClient([
            'Action: {"tool": "does.not.exist", "args": {}}',
            "Final: gave up on that tool",
        ])
        result = ReActAgent(client, registry).run("x")
        assert result.stop_reason == "final"
        assert result.steps[0].tool_result is not None
        assert result.steps[0].tool_result.ok is False
        assert "No such tool" in result.steps[0].tool_result.error

    def test_bad_arguments_are_reported_back_to_the_model(self, registry):
        client = ScriptedClient([
            'Action: {"tool": "echo.say", "args": {"wrong": "key"}}',
            "Final: done",
        ])
        result = ReActAgent(client, registry).run("x")
        error = result.steps[0].tool_result.error
        assert "missing required argument" in error
        assert "unknown argument" in error

    def test_step_budget_is_enforced(self, registry):
        client = ScriptedClient(
            ['Action: {"tool": "echo.say", "args": {"text": "loop"}}'] * 20
        )
        result = ReActAgent(client, registry, max_steps=3).run("loop forever")
        assert result.stop_reason == "max_steps"
        assert len(result.tool_calls) == 3

    def test_critical_tool_use_is_recorded(self, registry):
        """Phase 5's checkpoint keys off this flag; Phase 0 only records it."""
        client = ScriptedClient([
            'Action: {"tool": "danger.wipe", "args": {}}',
            "Final: wiped",
        ])
        result = ReActAgent(client, registry).run("wipe it")
        assert result.steps[0].tool_is_critical is True

    def test_bare_agent_does_not_block_a_critical_call(self, registry):
        """Condition A is unguarded by definition - this is the baseline failure."""
        client = ScriptedClient([
            'Action: {"tool": "danger.wipe", "args": {}}',
            "Final: wiped",
        ])
        result = ReActAgent(client, registry).run("wipe it")
        assert result.tool_calls == [("danger.wipe", {})]
        assert result.steps[0].tool_result.content == "wiped"

    def test_llm_error_is_captured_not_raised(self, registry):
        class Exploding:
            def chat(self, messages, **kwargs):
                raise RuntimeError("upstream is down")

        result = ReActAgent(Exploding(), registry).run("x")
        assert result.stop_reason == "error"
        assert "upstream is down" in result.error

    def test_cache_hits_are_counted(self, registry):
        class Cached(ScriptedClient):
            def chat(self, messages, **kwargs):
                response = super().chat(messages, **kwargs)
                return LLMResponse(
                    content=response.content, model_used="test/model",
                    latency_ms=0, from_cache=True, raw={},
                )

        result = ReActAgent(Cached(["Final: done"]), registry).run("x")
        assert result.num_llm_calls == 1
        assert result.num_cache_hits == 1


class TestAgentResult:
    def test_models_used_reports_a_fallback(self):
        result = AgentResult(task="t", final_answer="", stop_reason="final")
        assert result.models_used == []
