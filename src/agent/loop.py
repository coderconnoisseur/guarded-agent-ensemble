"""ReAct-style agent loop - Condition A, the bare backbone.

Implements the prompted-JSON protocol from CLAUDE.md 5.3: the model emits
`Thought:` plus either `Action: {json}` or `Final: <answer>`, this module
parses it deterministically, dispatches the tool, and feeds the result back
as an `Observation:`.

No defenses are attached here. This is the unguarded baseline the whole
project measures against, and Phase 1's `pipeline/condition_a.py` is a thin
wrapper over it. Phase 2 onward compose defense modules *around* this loop
rather than editing it, so Condition A and Condition B always share the same
backbone behaviour.

Everything the loop does is recorded on the returned `AgentResult`: the raw
model replies, each parse outcome, each tool result, and per-call LLM
provenance (model used, latency, cache hit). CLAUDE.md 2 requires that -
the eval harness and the Phase 4 flagship demo both need to show exactly what
happened, step by step.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from src.agent import prompts
from src.llm.client import LLMClient, LLMResponse
from src.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing (CLAUDE.md 5.3)
# ---------------------------------------------------------------------------

# Markers are matched at line start, tolerating surrounding markdown emphasis
# and leading whitespace, because small free-tier models bold their headers
# more often than not.
_ACTION_RE = re.compile(r"^[ \t>*_#]*action[ \t]*:?[ \t]*", re.IGNORECASE | re.MULTILINE)
_FINAL_RE = re.compile(
    r"^[ \t>*_#]*(?:final(?:[ \t]answer)?)[ \t]*:?[ \t]*", re.IGNORECASE | re.MULTILINE
)
_THOUGHT_RE = re.compile(
    r"^[ \t>*_#]*thought[ \t]*:?[ \t]*(.+?)(?=\n[ \t>*_#]*(?:action|final)\b|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?|```", re.MULTILINE)


@dataclass
class ParsedReply:
    """The outcome of parsing one model reply."""

    kind: str  # "action" | "final" | "unparseable"
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    final_text: str = ""
    problem: str = ""


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Pull the first balanced {...} object out of text.

    Brace-balanced rather than regex-based, and string-aware, so a JSON value
    that itself contains braces or quotes (an email body, a file's contents)
    does not truncate the payload at the wrong character.

    Returns (parsed_object, error_message).
    """
    start = text.find("{")
    if start == -1:
        return None, "no JSON object found after the Action marker"

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : index + 1]
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError as exc:
                    return None, f"the Action JSON is malformed ({exc.msg})"
                if not isinstance(parsed, dict):
                    return None, "the Action payload must be a JSON object"
                return parsed, ""
    return None, "the Action JSON object is not closed - a '}' is missing"


def parse_agent_reply(text: str) -> ParsedReply:
    """Parse one model reply into an action, a final answer, or a failure.

    Deliberately tolerant of formatting noise (code fences, bold markers) but
    strict about the payload: a malformed Action becomes an `unparseable`
    reply with a specific problem string, which the loop hands back to the
    model for one or two repair attempts before abandoning the step.
    """
    if not text or not text.strip():
        return ParsedReply(kind="unparseable", problem="the reply was empty")

    cleaned = _FENCE_RE.sub("", text)

    thought_match = _THOUGHT_RE.search(cleaned)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = _ACTION_RE.search(cleaned)
    final_match = _FINAL_RE.search(cleaned)

    # 5.3 says never both. If a model emits both anyway, honour whichever it
    # committed to first rather than silently dropping a tool call.
    if action_match and final_match:
        logger.warning("Reply contained both Action and Final; taking the earlier one.")
        if final_match.start() < action_match.start():
            action_match = None
        else:
            final_match = None

    if action_match:
        payload, error = _extract_json_object(cleaned[action_match.end() :])
        if payload is None:
            return ParsedReply(kind="unparseable", thought=thought, problem=error)

        tool = payload.get("tool") or payload.get("name") or payload.get("tool_name")
        if not isinstance(tool, str) or not tool.strip():
            return ParsedReply(
                kind="unparseable",
                thought=thought,
                problem='the Action JSON is missing a string "tool" key',
            )

        args = payload.get("args", payload.get("arguments", {}))
        if args is None:
            args = {}
        if isinstance(args, str):  # some models double-encode the args object
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return ParsedReply(
                    kind="unparseable",
                    thought=thought,
                    problem='"args" was a string that is not valid JSON',
                )
        if not isinstance(args, dict):
            return ParsedReply(
                kind="unparseable",
                thought=thought,
                problem='"args" must be a JSON object',
            )
        return ParsedReply(kind="action", thought=thought, tool=tool.strip(), args=args)

    if final_match:
        answer = cleaned[final_match.end() :].strip()
        if not answer:
            return ParsedReply(
                kind="unparseable",
                thought=thought,
                problem="the Final marker was present but the answer was empty",
            )
        return ParsedReply(kind="final", thought=thought, final_text=answer)

    return ParsedReply(
        kind="unparseable",
        thought=thought,
        problem="no 'Action:' or 'Final:' marker was present",
    )


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One LLM call and whatever followed from it."""

    index: int
    raw_reply: str
    parsed: ParsedReply
    model_used: str
    latency_ms: int
    from_cache: bool
    finish_reason: str = ""
    provider_filtered: bool = False
    tool_result: ToolResult | None = None
    # Registry metadata, recorded so later phases can see which steps *would*
    # have tripped a defense. Phase 0 attaches no behaviour to it.
    tool_is_critical: bool = False

    def summary(self) -> str:
        if self.parsed.kind == "action":
            status = "ok" if (self.tool_result and self.tool_result.ok) else "error"
            return f"action {self.parsed.tool} -> {status}"
        if self.parsed.kind == "final":
            return "final answer"
        return f"unparseable ({self.parsed.problem})"


@dataclass
class AgentResult:
    """Everything that happened during one `run()`."""

    task: str
    final_answer: str
    stop_reason: str  # "final" | "max_steps" | "parse_failure" | "error"
    steps: list[AgentStep] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    num_llm_calls: int = 0
    num_cache_hits: int = 0
    total_latency_ms: int = 0
    error: str | None = None

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Ordered (tool, args) pairs actually dispatched.

        The graders in Phase 1 onward check this rather than asking an LLM
        whether the agent misbehaved (CLAUDE.md 2, "determinism where possible").
        """
        return [
            (s.parsed.tool, s.parsed.args)
            for s in self.steps
            if s.parsed.kind == "action" and s.tool_result is not None
        ]

    @property
    def provider_filtered(self) -> bool:
        """Did a provider-side safety layer intervene on any step?

        If it did, a refusal recorded for this run is not evidence about the
        backbone - it is evidence about the provider's filter. The eval report
        separates the two rather than folding them together into HS.
        """
        return any(step.provider_filtered for step in self.steps)

    @property
    def finish_reasons(self) -> list[str]:
        return [step.finish_reason for step in self.steps if step.finish_reason]

    @property
    def models_used(self) -> list[str]:
        """Distinct backbone models, in order. Non-singleton means a fallback fired."""
        seen: list[str] = []
        for step in self.steps:
            if step.model_used and step.model_used not in seen:
                seen.append(step.model_used)
        return seen


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class ReActAgent:
    """Bare ReAct agent: no Harm Gate, no Planner, no Checkpoint, no Firewall."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        max_steps: int | None = None,
        max_parse_retries: int | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.client = client
        self.registry = registry
        self.max_steps = max_steps if max_steps is not None else settings.AGENT_MAX_STEPS
        self.max_parse_retries = (
            max_parse_retries
            if max_parse_retries is not None
            else settings.AGENT_MAX_PARSE_RETRIES
        )
        self.llm_kwargs = llm_kwargs

    def run(self, task: str) -> AgentResult:
        """Run one task to completion, a step budget, or a give-up."""
        system_prompt = prompts.build_system_prompt(self.registry.describe_for_prompt())
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        result = AgentResult(
            task=task,
            final_answer="",
            stop_reason="max_steps",
            transcript=[dict(m) for m in messages],
        )

        started = time.perf_counter()
        # One budget for tool-taking steps, plus a small allowance so format
        # repairs cannot starve the task of its real steps.
        max_llm_calls = self.max_steps + self.max_parse_retries
        consecutive_parse_failures = 0
        executed_steps = 0

        while result.num_llm_calls < max_llm_calls and executed_steps < self.max_steps:
            try:
                response = self.client.chat(messages, **self.llm_kwargs)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.error("LLM call failed: %s", exc)
                result.stop_reason = "error"
                result.error = f"{type(exc).__name__}: {exc}"
                break

            result.num_llm_calls += 1
            if response.from_cache:
                result.num_cache_hits += 1

            step = self._record_reply(result, response, messages)

            if step.parsed.kind == "unparseable":
                consecutive_parse_failures += 1
                if consecutive_parse_failures > self.max_parse_retries:
                    logger.warning("Giving up after %d parse failures", consecutive_parse_failures)
                    result.stop_reason = "parse_failure"
                    result.final_answer = response.content.strip()
                    break
                repair = prompts.build_repair_prompt(step.parsed.problem)
                messages.append({"role": "user", "content": repair})
                result.transcript.append({"role": "user", "content": repair})
                continue

            consecutive_parse_failures = 0

            if step.parsed.kind == "final":
                result.final_answer = step.parsed.final_text
                result.stop_reason = "final"
                break

            # An action: dispatch it and feed the observation back.
            executed_steps += 1
            tool_result = self.registry.dispatch(step.parsed.tool, step.parsed.args)
            step.tool_result = tool_result
            step.tool_is_critical = (
                self.registry.get(step.parsed.tool).critical
                if self.registry.has(step.parsed.tool)
                else False
            )
            logger.info(
                "Step %d: %s(%s) -> %s",
                step.index,
                step.parsed.tool,
                json.dumps(step.parsed.args, ensure_ascii=False)[:120],
                "ok" if tool_result.ok else f"error: {tool_result.error}",
            )

            observation = prompts.build_observation(tool_result.as_observation())
            messages.append({"role": "user", "content": observation})
            result.transcript.append(
                {"role": "tool", "content": tool_result.as_observation()}
            )

        if result.stop_reason == "max_steps" and not result.final_answer:
            logger.warning("Step budget exhausted before a Final answer")
            result.final_answer = (
                "(no final answer - the agent used its whole step budget)"
            )

        result.total_latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    @staticmethod
    def _record_reply(
        result: AgentResult, response: LLMResponse, messages: list[dict[str, str]]
    ) -> AgentStep:
        """Append the model's reply to both the message list and the transcript."""
        messages.append({"role": "assistant", "content": response.content})
        result.transcript.append({"role": "assistant", "content": response.content})

        step = AgentStep(
            index=len(result.steps) + 1,
            raw_reply=response.content,
            parsed=parse_agent_reply(response.content),
            model_used=response.model_used,
            latency_ms=response.latency_ms,
            from_cache=response.from_cache,
            finish_reason=response.finish_reason,
            provider_filtered=response.provider_filtered,
        )
        result.steps.append(step)
        return step
