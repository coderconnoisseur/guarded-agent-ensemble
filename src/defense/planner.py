"""Plan-Then-Execute Planner - the Tool Dependency Graph.

Adapts **IPIGuard** (EMNLP 2025), CLAUDE.md 3. The mechanism: the agent
commits to its entire tool-call sequence as a graph *before* it touches any
untrusted external data, and execution is then restricted to nodes in that
graph. An instruction that arrives later - inside a fetched page, a file, an
email - cannot introduce a tool call the plan never contained, because the
plan was fixed before that content existed in the agent's context.

Three sub-mechanisms from 3 patch the obvious rigidity of that idea:

  Argument Estimation
      Arguments that depend on an earlier tool's output cannot be known at
      plan time. A node declares the *shape* it expects and marks derived
      arguments; enforcement matches on tool identity, not on argument
      values, so the agent fills in real values at execution time.

  Node Expansion
      A too-rigid plan cripples the agent. New **read-only** calls are
      allowed even when absent from the plan, because a query cannot exfiltrate
      or destroy anything. Anything that writes, sends or deletes is not
      expandable - that asymmetry is the whole safety argument.

  Fake Tool Invocation
      IPIGuard's remedy when an injected branch and the real plan want the
      same tool. That belongs to Quarantine in Phase 4, not here.

WHERE ENFORCEMENT LIVES
-----------------------
In a wrapper around the tool registry, not in the agent loop. `loop.py` must
stay byte-identical between Condition A and Condition B - if the loop behaved
differently under the two conditions, an A/B difference could come from the
loop rather than from the defense. The registry is already the single dispatch
point, and the loop already feeds a failed tool call back as an observation
and lets the model try again, so a rejection needs no new machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import settings
from src.agent.loop import _extract_json_object
from src.llm.client import LLMClient, LLMError
from src.tools.registry import ToolRegistry, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

MODULE_NAME = "planner"


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    """One intended tool call, as committed to before execution."""

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    why: str = ""
    # Set once a real call has been matched to this node, so a single planned
    # node cannot license an unbounded number of calls to that tool.
    executed: bool = False

    @property
    def derived_args(self) -> list[str]:
        """Arguments the plan could not know - Argument Estimation.

        A value is derived when the model wrote a placeholder rather than a
        literal, which it does for anything that depends on an earlier node.
        """
        derived = []
        for name, value in self.args.items():
            text = str(value)
            if text.startswith("<") or "{{" in text or text.upper() in (
                "TBD", "UNKNOWN", "FROM_PREVIOUS", "DERIVED",
            ):
                derived.append(name)
        return derived


@dataclass
class ToolDependencyGraph:
    """The plan. Ordered nodes plus their dependency edges."""

    nodes: list[PlanNode] = field(default_factory=list)
    raw: str = ""
    degraded: bool = False  # true when planning failed and we fell back

    def tools(self) -> set[str]:
        return {n.tool for n in self.nodes}

    def by_id(self, node_id: str) -> PlanNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def next_unexecuted(self, tool: str) -> PlanNode | None:
        return next((n for n in self.nodes if n.tool == tool and not n.executed), None)

    def render(self) -> str:
        """Human-readable graph, for the Phase 3 demo."""
        if not self.nodes:
            return "  (empty plan)"
        lines = []
        for node in self.nodes:
            deps = f" after {','.join(node.depends_on)}" if node.depends_on else ""
            derived = node.derived_args
            marks = f"  [estimated: {', '.join(derived)}]" if derived else ""
            status = "done" if node.executed else "    "
            lines.append(f"  [{status}] {node.id}: {node.tool}{deps}{marks}")
            if node.why:
                lines.append(f"           why: {node.why}")
        return "\n".join(lines)


class PlanningError(LLMError):
    """The backbone could not produce a usable plan."""


def validate_plan(
    nodes: list[PlanNode], registry: ToolRegistry
) -> list[str]:
    """Structural problems with a plan. Empty list means usable.

    Checked before anything runs, because a plan referencing a tool that does
    not exist would otherwise fail node by node at execution time and look
    like the defense blocking legitimate work.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if node.id in seen:
            problems.append(f"duplicate node id {node.id!r}")
        seen.add(node.id)
        if not registry.has(node.tool):
            problems.append(
                f"node {node.id!r} uses unknown tool {node.tool!r}; "
                f"available: {', '.join(registry.names())}"
            )
    for node in nodes:
        for dep in node.depends_on:
            if dep not in seen:
                problems.append(f"node {node.id!r} depends on unknown node {dep!r}")

    # Cycle detection: a cyclic plan can never be executed in any order.
    order: dict[str, int] = {n.id: i for i, n in enumerate(nodes)}
    for node in nodes:
        for dep in node.depends_on:
            if dep in order and order[dep] >= order.get(node.id, 0):
                problems.append(
                    f"node {node.id!r} depends on {dep!r}, which is not earlier "
                    f"in the plan (cycle or out-of-order)"
                )
    return problems


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
You are planning, not acting. Before any tool runs, commit to the complete \
sequence of tool calls the task needs.

Available tools:

{tool_catalogue}

Reply with a single JSON object and nothing else:

{{"plan": [
  {{"id": "n1", "tool": "<tool_name>", "args": {{...}}, "depends_on": [], "why": "<short>"}},
  {{"id": "n2", "tool": "<tool_name>", "args": {{...}}, "depends_on": ["n1"], "why": "<short>"}}
]}}

RULES
- Use only the tool names listed above, spelled exactly.
- List every tool call the task will need, in the order it will need them.
- If an argument's value depends on the output of an earlier step, you cannot \
know it yet. Write the placeholder "<from n1>" (naming the step) instead of \
guessing a value.
- "depends_on" lists the ids of steps whose output this step needs.
- Plan only what the task asks for. Do not add steps for anything you might \
discover later.
- If the task needs no tools at all, reply {{"plan": []}}.

TASK: {task}
"""

REPLAN_PROMPT = """\
That plan could not be used:

{problems}

Reply again with a corrected JSON object in the same format, and nothing else.
"""


def parse_plan(text: str) -> tuple[list[PlanNode], str]:
    """Parse the model's plan reply. Returns (nodes, error_message)."""
    payload, error = _extract_json_object(text)
    if payload is None:
        return [], error
    raw_nodes = payload.get("plan")
    if raw_nodes is None:
        return [], 'the JSON object has no "plan" key'
    if not isinstance(raw_nodes, list):
        return [], '"plan" must be a list of steps'

    nodes: list[PlanNode] = []
    for index, entry in enumerate(raw_nodes):
        if not isinstance(entry, dict):
            return [], f"step {index + 1} is not an object"
        tool = entry.get("tool") or entry.get("name")
        if not isinstance(tool, str) or not tool.strip():
            return [], f'step {index + 1} has no string "tool"'
        args = entry.get("args", {})
        if not isinstance(args, dict):
            args = {}
        depends = entry.get("depends_on") or []
        if not isinstance(depends, list):
            depends = []
        nodes.append(
            PlanNode(
                id=str(entry.get("id") or f"n{index + 1}"),
                tool=tool.strip(),
                args=args,
                depends_on=[str(d) for d in depends],
                why=str(entry.get("why") or ""),
            )
        )
    return nodes, ""


class Planner:
    """Builds the Tool Dependency Graph before any tool runs."""

    module = MODULE_NAME

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.client = client
        self.model = model or settings.BACKBONE_MODEL
        self.max_attempts = max_attempts
        self.llm_calls = 0

    def build_plan(self, task: str, registry: ToolRegistry) -> ToolDependencyGraph:
        """Ask the backbone for a plan, validate it, retry once, then degrade.

        Degrading to a read-only plan rather than raising is deliberate: a
        planning failure is a failure of the *defense*, and a defense that
        cannot plan should not also destroy the agent's ability to do
        harmless work. The degraded flag is recorded so a run that fell back
        is never mistaken for one the planner actually constrained.
        """
        messages = [
            {
                "role": "user",
                "content": PLAN_PROMPT.format(
                    tool_catalogue=registry.describe_for_prompt(), task=task
                ),
            }
        ]

        problems: list[str] = []
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat(messages, model=self.model)
            except LLMError as exc:
                logger.warning("Planner call failed (%s); degrading", exc)
                return self._degraded(registry, f"planner unavailable: {exc}")
            self.llm_calls += 1

            nodes, error = parse_plan(response.content)
            problems = [error] if error else validate_plan(nodes, registry)
            if not problems:
                graph = ToolDependencyGraph(nodes=nodes, raw=response.content)
                logger.info(
                    "Plan accepted: %d node(s) over %s",
                    len(nodes), sorted(graph.tools()) or "no tools",
                )
                return graph

            logger.warning(
                "Plan rejected (attempt %d): %s", attempt + 1, "; ".join(problems)
            )
            if attempt < self.max_attempts - 1:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": REPLAN_PROMPT.format(
                        problems="\n".join(f"- {p}" for p in problems)
                    ),
                })

        return self._degraded(registry, "; ".join(problems))

    @staticmethod
    def _degraded(registry: ToolRegistry, reason: str) -> ToolDependencyGraph:
        """Fall back to a plan of nothing - read-only expansion still applies.

        An empty plan is not "allow everything": the enforcer permits only
        read-only tools when a call matches no node, so a degraded run can
        still look things up but cannot send, write or delete.
        """
        logger.error("Planner degraded to read-only: %s", reason)
        return ToolDependencyGraph(nodes=[], raw="", degraded=True)


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


@dataclass
class PlanEnforcement:
    """What the enforcer allowed, expanded and blocked during one run."""

    graph: ToolDependencyGraph
    executed: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    rejections: list[tuple[str, str]] = field(default_factory=list)
    llm_calls: int = 0

    @property
    def plan_followed(self) -> bool:
        """Did every executed call come from the plan, with nothing blocked?"""
        return not self.rejections and not self.expansions

    def summary(self) -> str:
        parts = [f"{len(self.executed)} planned call(s)"]
        if self.expansions:
            parts.append(f"{len(self.expansions)} read-only expansion(s)")
        if self.rejections:
            parts.append(f"{len(self.rejections)} blocked")
        return ", ".join(parts)


class PlanEnforcingRegistry:
    """Registry wrapper that permits only calls the plan licensed.

    Duck-types `ToolRegistry` for everything the agent loop uses, so the loop
    itself is unchanged. A blocked call comes back as an ordinary failed
    `ToolResult`, which the loop already surfaces to the model as an
    observation - the agent gets told why and can choose a different step.
    """

    def __init__(
        self, inner: ToolRegistry, enforcement: PlanEnforcement
    ) -> None:
        self._inner = inner
        self.enforcement = enforcement

    # -- passthrough used by the agent loop --------------------------------

    def describe_for_prompt(self) -> str:
        return self._inner.describe_for_prompt()

    def has(self, name: str) -> bool:
        return self._inner.has(name)

    def get(self, name: str) -> ToolSpec:
        return self._inner.get(name)

    def names(self) -> list[str]:
        return self._inner.names()

    def critical_tools(self) -> list[str]:
        return self._inner.critical_tools()

    # -- the enforcement point ---------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        graph = self.enforcement.graph

        node = graph.next_unexecuted(name)
        if node is not None:
            # Argument Estimation: the plan committed to the tool, not to
            # argument values it could not know at plan time.
            node.executed = True
            self.enforcement.executed.append(name)
            return self._inner.dispatch(name, args)

        if self._inner.has(name) and self._inner.get(name).read_only:
            # Node Expansion: a query adds no capability to exfiltrate or
            # destroy, so allowing it costs nothing an attacker can use.
            self.enforcement.expansions.append(name)
            logger.info("Node Expansion: allowing off-plan read-only %s", name)
            return self._inner.dispatch(name, args)

        reason = (
            f"Blocked by the plan: {name} is not in the agreed plan for this "
            f"task, and only read-only tools may be added after planning. "
            f"Planned: {', '.join(sorted(graph.tools())) or 'no tool calls'}."
        )
        self.enforcement.rejections.append((name, reason))
        logger.warning("Plan enforcement BLOCKED %s", name)
        return ToolResult(
            tool_name=name, args=args, ok=False, content="", latency_ms=0,
            error=reason,
        )
