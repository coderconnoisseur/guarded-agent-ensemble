"""Condition B - the backbone wrapped in the ensemble's defense modules.

Sibling of `condition_a.py`, same `run(task) -> AgentResult` interface, so the
eval runner cannot tell them apart and neither condition gets a different code
path to the graders.

`enabled_modules` is the whole point of the class (CLAUDE.md 9.1 tier 2).
Passing a subset runs exactly those defenses, which is what makes two claims
measurable:

  - the **cumulative ablation** (9.1 tier 1): each phase snapshots whatever
    is wired in so far, giving five configurations by Phase 6.
  - the **single-module isolation** (9.1 tier 2): each module alone against
    the full mixed suite, which is what shows each one covering its own
    threat model and nothing else - the actual argument for an *ensemble*.

Phases 2 and 3 wire the first two modules. The remaining names are declared
but not yet implemented; asking for one raises rather than silently running
unguarded,
because a config typo that quietly disables a defense would show up as a
suspiciously good number rather than an error.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import settings
from src.agent.loop import AgentResult, ReActAgent
from src.defense.firewall import (
    FirewallRegistry,
    ResponseFirewall,
    scan_tool_descriptions,
)
from src.defense.harm_gate import HarmGate, HarmVerdict
from src.defense.misalignment import MisalignmentCheckpoint, MisalignmentRegistry
from src.defense.quarantine import Quarantine
from src.defense.planner import (
    PlanEnforcement,
    PlanEnforcingRegistry,
    Planner,
)
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry, build_default_registry, build_registry

logger = logging.getLogger(__name__)

CONDITION = "B"

# All four architecture.md defense nodes are implemented as of Phase 5.
IMPLEMENTED_MODULES: frozenset[str] = frozenset(
    {"harm_gate", "planner", "firewall", "quarantine", "misalignment"}
)
ALL_MODULES: frozenset[str] = frozenset(
    {"harm_gate", "planner", "firewall", "quarantine", "misalignment"}
)


class ConditionB:
    """The guarded pipeline. Composes defense modules around the ReAct loop."""

    condition = CONDITION

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry | None = None,
        enabled_modules: set[str] | None = None,
        model: str | None = None,
        use_chain: bool = False,
        enable_harm_classifier: bool = True,
        enable_guard_model: bool = True,
        **agent_kwargs: Any,
    ) -> None:
        requested = frozenset(
            enabled_modules if enabled_modules is not None else IMPLEMENTED_MODULES
        )
        unknown = requested - ALL_MODULES
        if unknown:
            raise ValueError(f"Unknown module(s): {sorted(unknown)}")
        unbuilt = requested - IMPLEMENTED_MODULES
        if unbuilt:
            raise NotImplementedError(
                f"Module(s) not implemented yet: {sorted(unbuilt)}. "
                f"Available now: {sorted(IMPLEMENTED_MODULES)}."
            )

        self.client = client
        self.registry = registry if registry is not None else build_default_registry()
        self.enabled_modules = requested

        # Same pinning rule as Condition A, and it matters more here: an A/B
        # comparison across two different backbones measures the models.
        self.model = model or (None if use_chain else settings.BACKBONE_MODEL)
        if self.model:
            agent_kwargs.setdefault("model", self.model)
        self.agent = ReActAgent(self.client, self.registry, **agent_kwargs)

        self.harm_gate = (
            HarmGate(client=client, enable_classifier=enable_harm_classifier)
            if "harm_gate" in self.enabled_modules
            else None
        )
        self.planner = (
            Planner(client=client, model=self.model)
            if "planner" in self.enabled_modules
            else None
        )
        self.checkpoint = (
            MisalignmentCheckpoint(client=client, model=self.model)
            if "misalignment" in self.enabled_modules
            else None
        )
        # Built once and reset per run: it accumulates the action/observation
        # trajectory InferAct's Task Inference Unit reasons over.
        self.misalignment_registry = (
            MisalignmentRegistry(self.registry, self.checkpoint)
            if self.checkpoint is not None
            else None
        )
        self.firewall = (
            ResponseFirewall(client=client, enable_guard_model=enable_guard_model)
            if "firewall" in self.enabled_modules
            else None
        )
        # Quarantine is the firewall's remedy; without the firewall there is
        # nothing to remediate, so asking for it alone is a configuration
        # error rather than a silently-inert module.
        if "quarantine" in self.enabled_modules and self.firewall is None:
            raise ValueError(
                "quarantine requires firewall: it remediates what the firewall "
                "flags, and alone it would never run."
            )
        self.quarantine = (
            Quarantine(self.firewall)
            if "quarantine" in self.enabled_modules and self.firewall is not None
            else None
        )

        if self.firewall is not None:
            # ShieldMCP Stage 1, once at wiring time: the registry itself can
            # be the attack surface.
            self.registry_integrity = scan_tool_descriptions(self.registry)
        else:
            self.registry_integrity = []

    def use_scenario(self, scenario: str) -> None:
        """Switch to a task scenario's tool surface (docs/HANDOFF.md 5.2).

        The misalignment wrapper has to be rebuilt, not just re-pointed: it is
        constructed around a registry, and leaving it wrapping the previous
        scenario's would have it guarding tools this case cannot call while
        waving through the ones it can. The plan and firewall wrappers are
        built per run, so they pick the new surface up on their own.
        """
        self.registry = build_registry(scenario)
        if self.checkpoint is not None:
            self.misalignment_registry = MisalignmentRegistry(
                self.registry, self.checkpoint
            )
        if self.firewall is not None:
            # ShieldMCP Stage 1 is per surface: a new scenario's descriptions
            # have not been integrity-checked by the scan done at wiring time.
            self.registry_integrity = scan_tool_descriptions(self.registry)
        self.agent.registry = self.registry

    # -- module: Harm Gate -------------------------------------------------

    @staticmethod
    def _blocked_result(task: str, verdict: HarmVerdict, elapsed_ms: int) -> AgentResult:
        """Synthesise the result of a task the gate refused.

        Shaped exactly like a real run so the graders need no special case:
        a refusal with no tool calls, which `detect_refusal` scores as a
        refusal on both of its signals. `num_llm_calls` counts only what the
        gate itself spent - the backbone was never invoked, which is the
        cheapness claim in architecture.md's Flow 3.
        """
        answer = verdict.refusal_text()
        return AgentResult(
            task=task,
            final_answer=answer,
            stop_reason="blocked_by_harm_gate",
            steps=[],
            transcript=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": answer},
            ],
            num_llm_calls=verdict.llm_calls,
            total_latency_ms=elapsed_ms,
        )

    # -- pipeline ----------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Run one task through every enabled defense, then the agent."""
        started = time.perf_counter()

        if self.harm_gate is not None:
            verdict = self.harm_gate.check(task)
            if verdict.flagged:
                elapsed = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "Condition B: Harm Gate blocked at %s stage (%s) after %dms",
                    verdict.stage, verdict.category, elapsed,
                )
                result = self._blocked_result(task, verdict, elapsed)
                result.harm_gate_verdict = verdict
                return result
            logger.debug("Harm Gate passed: %s", verdict.reason)

        # IPIGuard: plan the whole tool sequence before the agent sees any
        # tool output, then restrict execution to that plan. Building the plan
        # first is the load-bearing part - a plan written after untrusted
        # content is in context is not a constraint on anything.
        enforcement: PlanEnforcement | None = None
        registry: Any = self.registry

        # InferAct: the ToM check sits innermost, so the plan gate above it
        # runs first. architecture.md Flow 4 orders the pipeline
        # `Planner -> Backbone -> Misalignment Checkpoint -> Tool`, and
        # dispatch enters at the outermost wrapper - so "innermost" here is
        # what puts the checkpoint *after* the plan gate at run time. It also
        # means a call the plan already rejects never spends two judge calls.
        if self.misalignment_registry is not None:
            self.misalignment_registry.reset(task)
            registry = self.misalignment_registry

        if self.planner is not None:
            graph = self.planner.build_plan(task, self.registry)
            enforcement = PlanEnforcement(
                graph=graph, llm_calls=self.planner.llm_calls
            )
            registry = PlanEnforcingRegistry(registry, enforcement)

        # Layered deliberately: the Planner decides whether a call may happen
        # (pre-call), the firewall decides whether its response may be
        # believed (post-response). Separate wrappers keep either runnable
        # alone for 9.1's single-module isolation.
        if self.firewall is not None:
            self.firewall.verdicts.clear()
            if self.quarantine is not None:
                self.quarantine.events.clear()
            registry = FirewallRegistry(registry, self.firewall, self.quarantine)

        self.agent.registry = registry
        result = self.agent.run(task)

        # Charge every defense's own calls to the run, so LAT and the call
        # counts include the cost of the defense rather than hiding it.
        if self.harm_gate is not None:
            result.num_llm_calls += verdict.llm_calls
            result.harm_gate_verdict = verdict
        if enforcement is not None:
            result.num_llm_calls += enforcement.llm_calls
            result.plan_enforcement = enforcement
        if self.firewall is not None:
            result.num_llm_calls += sum(v.llm_calls for v in self.firewall.verdicts)
            result.firewall_verdicts = list(self.firewall.verdicts)
        if self.quarantine is not None:
            result.quarantine_events = list(self.quarantine.events)
        if self.misalignment_registry is not None:
            verdicts = list(self.misalignment_registry.verdicts)
            result.num_llm_calls += sum(v.llm_calls for v in verdicts)
            result.misalignment_verdicts = verdicts
        return result
