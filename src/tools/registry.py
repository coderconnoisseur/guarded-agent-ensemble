"""Vetted Tool Registry - tool schemas plus the dispatch table.

This is the "Vetted Tool Registry" node in docs/architecture.md: the single
store of tool descriptions the planner and the agent loop are allowed to see.

Phase 0 builds the schemas, the dispatch table, and the *metadata* that later
defense modules key off. The modules themselves are later phases:

  - `critical`          -> InferAct's Misalignment Checkpoint fires only at
                           critical actions (CLAUDE.md 3, Phase 5).
  - `read_only`         -> IPIGuard's Node Expansion may add read-only query
                           calls to a plan, nothing else (CLAUDE.md 3, Phase 3).
  - `returns_untrusted` -> ShieldMCP's Stage 3 scans responses that carry
                           external data (CLAUDE.md 3, Phase 4).
  - `description_sha256` -> anchor for ShieldMCP's Stage 1 integrity check on
                           tool descriptions (CLAUDE.md 3, Phase 4).

No defense logic lives here yet - these are the hooks those phases hang off.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., str]


@dataclass(frozen=True)
class ToolParam:
    """One declared argument of a tool."""

    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class ToolSpec:
    """A tool's schema, handler, and defense-relevant metadata."""

    name: str
    description: str
    params: list[ToolParam]
    handler: ToolHandler

    # Irreversible or high-impact. Phase 5's ToM checkpoint triggers on these.
    critical: bool = False
    # Pure query, no state change. Phase 3's Node Expansion may add these.
    read_only: bool = True
    # Returns data from outside the trust boundary. Phase 4's firewall scans it.
    returns_untrusted: bool = False

    description_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.description.encode("utf-8")).hexdigest()
        object.__setattr__(self, "description_sha256", digest)

    def signature(self) -> str:
        """Human/LLM-readable one-liner, e.g. `files.read(path: string)`."""
        rendered = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else " (optional)")
            for p in self.params
        )
        return f"{self.name}({rendered})"

    def describe(self) -> str:
        """Full schema block for the system prompt."""
        lines = [f"- {self.signature()}", f"    {self.description}"]
        for p in self.params:
            flag = "" if p.required else " [optional]"
            lines.append(f"    - {p.name} ({p.type}){flag}: {p.description}")
        tags = []
        if self.critical:
            tags.append("CRITICAL - irreversible or high-impact")
        if not self.read_only:
            tags.append("modifies state")
        if tags:
            lines.append(f"    ! {'; '.join(tags)}")
        return "\n".join(lines)


@dataclass
class ToolResult:
    """Outcome of one dispatch, recorded verbatim in the transcript."""

    tool_name: str
    args: dict[str, Any]
    ok: bool
    content: str
    latency_ms: int
    error: str | None = None

    def as_observation(self) -> str:
        """How this result is fed back into the agent's context."""
        if self.ok:
            return self.content
        return f"ERROR: {self.error}"


class ToolNotFoundError(KeyError):
    """Requested tool is not in the registry."""


class ToolArgumentError(ValueError):
    """Arguments did not match the declared schema."""


class ToolRegistry:
    """Holds every tool the agent may call, and dispatches to them."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        logger.debug(
            "Registered tool %s (critical=%s, read_only=%s, untrusted=%s)",
            spec.name,
            spec.critical,
            spec.read_only,
            spec.returns_untrusted,
        )
        return spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    def critical_tools(self) -> list[str]:
        """Names the Phase 5 checkpoint will guard."""
        return [s.name for s in self.specs() if s.critical]

    def describe_for_prompt(self) -> str:
        """The tool catalogue block injected into the system prompt."""
        return "\n".join(spec.describe() for spec in self.specs())

    def validate_args(self, name: str, args: dict[str, Any]) -> None:
        """Check args against the declared schema.

        Raises ToolArgumentError with a message aimed at the *model*, since the
        agent loop feeds it straight back as an observation for a retry.
        """
        spec = self.get(name)
        declared = {p.name for p in spec.params}
        missing = [p.name for p in spec.params if p.required and p.name not in args]
        unknown = sorted(set(args) - declared)
        problems = []
        if missing:
            problems.append(f"missing required argument(s): {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown argument(s): {', '.join(unknown)}")
        if problems:
            raise ToolArgumentError(
                f"{'; '.join(problems)}. Expected signature: {spec.signature()}"
            )

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run a tool and wrap the outcome. Never raises for tool-level errors.

        A failing tool is an observation the agent should get a chance to react
        to, not a crash - so everything short of a programming bug comes back
        as `ok=False` with an error string.
        """
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        if not self.has(name):
            return ToolResult(
                tool_name=name,
                args=args,
                ok=False,
                content="",
                latency_ms=elapsed(),
                error=f"No such tool: {name}. Available tools: {', '.join(self.names())}",
            )

        try:
            self.validate_args(name, args)
        except ToolArgumentError as exc:
            return ToolResult(
                tool_name=name, args=args, ok=False, content="",
                latency_ms=elapsed(), error=str(exc),
            )

        spec = self.get(name)
        try:
            content = spec.handler(**args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent, not swallowed
            logger.warning("Tool %s raised: %s", name, exc)
            return ToolResult(
                tool_name=name, args=args, ok=False, content="",
                latency_ms=elapsed(), error=f"{type(exc).__name__}: {exc}",
            )

        return ToolResult(
            tool_name=name, args=args, ok=True,
            content=str(content), latency_ms=elapsed(),
        )


def build_default_registry() -> ToolRegistry:
    """The standard tool set: sandboxed files, stubbed web, mocked comms.

    Imported lazily so `registry.py` stays importable without the tool modules
    (the unit tests build registries with fake tools).
    """
    from src.tools import comms, files, web

    registry = ToolRegistry()
    for module in (files, web, comms):
        for spec in module.TOOLS:
            registry.register(spec)
    return registry
