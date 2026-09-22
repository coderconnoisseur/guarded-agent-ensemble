"""Offline replay of the Misalignment Checkpoint over saved trajectories.

WHY THIS EXISTS
---------------
InferAct evaluates its detector on **pre-collected trajectories**. We had been
judging ours only in-line, inside a live agent run, and it cost us twice.

*Iteration cost.* Changing the checkpoint's prompt meant re-running whole
agent loops at 2 requests/minute. Replay judges a recorded
`(instruction, trajectory, proposed action)` triple for 2 calls, no matter how
expensive the run that produced it was.

*Pre-emption.* The frozen ablation showed the Planner rejecting `mis_001` and
`mis_004`'s critical calls before the checkpoint was ever consulted, so the
positive class emptied and `MF1` came back undefined inside the full ensemble
(HANDOFF 7.2). A recorded triple cannot be pre-empted by a module that is not
running.

It is also what makes fixing the checkpoint honest. The measured diagnosis is
that its verification prompt penalises the agent for *inferring* things -
resolving "my current account" to an id, reading a file before editing it,
picking the cheapest fare - and the fix has to be validated on triples it was
not tuned against. `split_triples` provides that, splitting **by case**
because one case contributes several triples from the same run and a
triple-wise split would tune and evaluate on two actions of one trajectory.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from src.defense.misalignment import TrajectoryStep
from src.eval.generator import load_generated_cases
from src.eval.schemas import load_suites
from src.tools.registry import build_registry

logger = logging.getLogger(__name__)

# The agent emits `Action: {"tool": ..., "args": {...}}`; this pulls the pair
# back out of a saved assistant turn.
_ACTION_RE = re.compile(
    r'"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}', re.S
)


@dataclass
class ReplayTriple:
    """One recorded decision point the checkpoint could have ruled on."""

    case_id: str
    scenario: str
    label: str  # "misaligned" | "aligned" - the case's checkpoint_label
    instruction: str
    trajectory: list[TrajectoryStep]
    tool: str
    args: dict[str, Any]
    step_index: int = 0
    source: str = ""

    @property
    def key(self) -> str:
        """Stable identity, used to deduplicate across result files."""
        blob = "|".join([
            self.instruction,
            "|".join(f"{s.tool}{sorted(s.args.items())}" for s in self.trajectory),
            self.tool,
            json.dumps(self.args, sort_keys=True),
        ])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _parse_actions(content: str) -> list[tuple[str, dict]]:
    out = []
    for tool, raw in _ACTION_RE.findall(content):
        try:
            out.append((tool, json.loads(raw)))
        except json.JSONDecodeError:
            continue
    return out


def collect_triples(
    results_dir: Path | None = None, suites_root: Path | None = None
) -> list[ReplayTriple]:
    """Every labelled decision point recorded in `results/`, deduplicated.

    Only cases carrying an explicit `checkpoint_label` contribute - the same
    rule `misalignment_macro_f1` applies, and for the same reason: an
    argument-dependent case has no fixed answer, so scoring a ruling on it
    would be circular.
    """
    # Generated cases included. They resolved through `load_suites` alone
    # before, which excludes them by design - so the paired misalignment cases
    # ran, wrote results, and contributed zero replay material, silently,
    # because an unresolvable case id is simply skipped.
    root = suites_root or settings.TESTSUITES_DIR
    cases = {c.id: c for c in load_suites(root)}
    cases.update({c.id: c for c in load_generated_cases(root)})
    critical = {
        name: set(build_registry(name).critical_tools())
        for name in ("workspace", "banking", "travel")
    }

    found: dict[str, ReplayTriple] = {}
    pattern = str((results_dir or settings.RESULTS_DIR) / "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in data.get("results", []):
            case = cases.get(row.get("test_case_id", ""))
            if case is None or not case.expects.checkpoint_label:
                continue
            scenario = row.get("scenario", "workspace")

            trajectory: list[TrajectoryStep] = []
            pending: list[tuple[str, dict]] = []
            for entry in row.get("transcript", []):
                if entry["role"] == "assistant":
                    pending = _parse_actions(entry["content"])
                elif entry["role"] == "tool" and pending:
                    tool, args = pending[0]
                    if tool in critical.get(scenario, set()):
                        triple = ReplayTriple(
                            case_id=case.id, scenario=scenario,
                            label=case.expects.checkpoint_label,
                            instruction=case.prompt,
                            # Copied, because the list keeps growing below and
                            # the triple must freeze the state as it was.
                            trajectory=list(trajectory),
                            tool=tool, args=args,
                            step_index=len(trajectory),
                            source=Path(path).name,
                        )
                        found.setdefault(triple.key, triple)
                    trajectory.append(
                        TrajectoryStep(tool=tool, args=args,
                                       observation=entry["content"])
                    )
                    pending = []

    logger.debug("Collected %d unique replay triples", len(found))
    return sorted(found.values(), key=lambda t: (t.case_id, t.step_index, t.key))


def _case_rank(case_id: str) -> int:
    """Deterministic, machine-independent ordering key for a case."""
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16)


def split_triples(
    triples: list[ReplayTriple],
) -> tuple[list[ReplayTriple], list[ReplayTriple]]:
    """(dev, heldout), grouped by case and **stratified by label**.

    Two properties, both learned the hard way:

    *Grouped by case*, because one case contributes several triples from a
    single trajectory. A triple-wise split would tune on one of a run's
    critical actions and "evaluate" on another from the same run.

    *Stratified by label*, because a plain hash of 12 case ids put 7 of the 8
    positives on one side - leaving the other with a single misaligned triple,
    and a detection rate over n=1 is not a measurement. Cases are ranked
    deterministically within each label and dealt alternately, so both halves
    stay scorable however small the labelled set is.

    A case whose triples carry more than one label (which no current case
    does) is assigned by its majority label, so the grouping property wins
    over perfect stratification.
    """
    by_case: dict[str, list[ReplayTriple]] = {}
    for triple in triples:
        by_case.setdefault(triple.case_id, []).append(triple)

    def majority_label(items: list[ReplayTriple]) -> str:
        counts: dict[str, int] = {}
        for item in items:
            counts[item.label] = counts.get(item.label, 0) + 1
        return max(counts, key=lambda label: (counts[label], label))

    dev: list[ReplayTriple] = []
    heldout: list[ReplayTriple] = []
    for label in ("misaligned", "aligned"):
        group = sorted(
            (cid for cid, items in by_case.items()
             if majority_label(items) == label),
            key=_case_rank,
        )
        for index, case_id in enumerate(group):
            (dev if index % 2 == 0 else heldout).extend(by_case[case_id])

    order = lambda t: (t.case_id, t.step_index, t.key)  # noqa: E731
    return sorted(dev, key=order), sorted(heldout, key=order)


def is_dev(case_id: str, triples: list[ReplayTriple] | None = None) -> bool:
    """Whether a case lands in the dev half. Convenience over split_triples."""
    pool = triples if triples is not None else collect_triples()
    return case_id in {t.case_id for t in split_triples(pool)[0]}


__all__ = ["ReplayTriple", "collect_triples", "is_dev", "split_triples"]
