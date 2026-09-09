"""Condition A - the bare backbone, no defenses attached.

This is the baseline every Condition B number is measured against, and it is
deliberately thin: 10 of CLAUDE.md predicted "no changes needed if Phase 0 was
built cleanly", and that held. All it adds over `ReActAgent` is a stable
entry point the eval runner can call without knowing how the agent is wired.

`enabled_modules` is accepted here even though Condition A never enables
anything. 9.1 tier 2 asks for that parameter to exist while the composition
code is still simple, so Phase 2's `condition_b.py` is a drop-in sibling with
the same signature rather than a retrofit. Passing a non-empty set here is an
error, not a silent no-op - a caller who does it has confused the conditions,
and quietly running unguarded would corrupt the baseline.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.loop import AgentResult, ReActAgent
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)

CONDITION = "A"

# Every module the ensemble will eventually offer. Condition A supports none
# of them; Condition B (Phase 2+) will accept any subset.
ALL_MODULES: frozenset[str] = frozenset(
    {"harm_gate", "planner", "firewall", "quarantine", "misalignment"}
)


class ConditionA:
    """Runs a task through the unguarded ReAct loop."""

    condition = CONDITION

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry | None = None,
        enabled_modules: set[str] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        if enabled_modules:
            raise ValueError(
                f"Condition A is the unguarded baseline and cannot enable modules; "
                f"got {sorted(enabled_modules)}. Use ConditionB for guarded runs."
            )
        self.client = client
        self.registry = registry if registry is not None else build_default_registry()
        self.enabled_modules: frozenset[str] = frozenset()
        self.agent = ReActAgent(self.client, self.registry, **agent_kwargs)

    def run(self, task: str) -> AgentResult:
        """Execute one task and return the full, inspectable result."""
        logger.debug("Condition A running task: %.80s", task)
        return self.agent.run(task)
