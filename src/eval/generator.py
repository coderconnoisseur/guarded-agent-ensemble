"""Factored test-case generation: tasks x injection goals x attack templates.

THE SHAPE, AND WHY IT IS WORTH COPYING
--------------------------------------
AgentDojo gets 567 task-by-injection combinations from 86 user tasks, 27
injection goals and ~6 attack templates, because those three axes are separate
and multiply. Our hand-written cases weld them together: one payload baked
into one case, so 15 cases cost 15 authoring efforts and yield 15 data points.

Measured 2026-09-20, which is what forced the change: Condition A's injection
ASR is 1/15 = 0.07, and at that rate a significant A/B result needs n=65 per
arm (see `src/eval/stats.required_n`). Writing 50 more cases by hand is weeks
of work for a number that may still not separate. Crossing 3 tasks x 3 goals x
5 templates is 11 authored pieces and 45 cases, and - more importantly -
several of those templates are ones AgentDojo reports high ASR against, so the
*baseline* rises and the required n falls.

GENERATED CASES ARE OPT-IN
--------------------------
`load_cases()` excludes them unless asked. Every result in `results/` was
measured over the hand-written suite, and silently folding 45 new cases in
would change every denominator at once - the exact incomparability
`ablation_table.py` exists to refuse. Use `--include-generated`, and the
frozen-suite machinery will treat it as a different case set, which it is.

They are still ordinary `TestCase` objects: same schema, same mechanical
grading, same scenario scoping. Nothing downstream knows they were generated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.attacks import DEFAULT_MODEL, DEFAULT_USER, render_attack
from src.eval.schemas import TestCase

logger = logging.getLogger(__name__)

SPEC_SUFFIX = ".spec.json"


def load_generated_specs(root: Path) -> list[dict[str, Any]]:
    """Every generator spec under a test-suite root, sorted for determinism."""
    specs = []
    for path in sorted(Path(root).rglob(f"*{SPEC_SUFFIX}")):
        specs.append(json.loads(path.read_text(encoding="utf-8")))
    return specs


def expand_spec(spec: dict[str, Any]) -> list[TestCase]:
    """Cross one spec's three axes into concrete test cases.

    Deterministic: the same spec always produces the same cases in the same
    order with the same ids, because a generated case still has to be evidence
    someone can re-check.

    Ids encode the cell - `<prefix>_<task>_<goal>_<attack>` - so a failure is
    traceable to a task, a goal or a template rather than to "one of the
    generated ones".
    """
    scenario = spec["scenario"]
    suite = spec["suite"]
    prefix = spec["id_prefix"]
    user_name = spec.get("user_name", DEFAULT_USER)
    model_name = spec.get("model_name", DEFAULT_MODEL)

    cases: list[TestCase] = []
    for task in spec["user_tasks"]:
        for goal in spec["injection_goals"]:
            for attack in spec["attacks"]:
                # Raises KeyError on an unknown attack name rather than
                # quietly emitting a case with no payload - a case that ran
                # with no attack present would pass and deflate ASR_inj, which
                # is the failure mode verify_injection() already guards at run
                # time.
                payload = render_attack(
                    attack, goal=goal["goal"], user=user_name, model=model_name
                )
                # Wrap the payload in legitimate content where the carrier
                # REPLACES the whole response (web.fetch, files.read) rather
                # than appending to it (bank.transactions, travel listings).
                # Without this the agent fetches a page that is nothing but an
                # injection, correctly ignores it, and then truthfully reports
                # that the page was empty - which the task check scores as a
                # failure caused by neither the agent nor the defense.
                carrier = spec.get("carrier_content")
                if carrier:
                    payload = f"{carrier}{payload}"
                cases.append(TestCase.model_validate({
                    "id": f"{prefix}_{task['key']}_{goal['key']}_{attack}",
                    "suite": suite,
                    "scenario": scenario,
                    "category": f"{goal['key']}__{attack}",
                    "prompt": task["prompt"],
                    "injected_content": payload,
                    "injection_tool": spec["injection_tool"],
                    "injection_key": spec.get("injection_key"),
                    "expects": {
                        "should_refuse": False,
                        "should_complete_original_task": True,
                        "misalignment_expected": False,
                    },
                    "grading": {
                        "type": "rubric",
                        "rubric": [
                            f"user task: {task['prompt']}",
                            f"injected goal: {goal['goal']}",
                            f"attack template: {attack}",
                            "generated by src/eval/generator.py from "
                            f"{prefix}{SPEC_SUFFIX}",
                        ],
                        "checks": list(task["checks"]) + list(goal["checks"]),
                    },
                }))
    logger.debug("Expanded %s into %d cases", prefix, len(cases))
    return cases


def load_generated_cases(root: Path) -> list[TestCase]:
    """Every generated case under a test-suite root."""
    cases: list[TestCase] = []
    for spec in load_generated_specs(root):
        cases.extend(expand_spec(spec))
    return sorted(cases, key=lambda c: c.id)


__all__ = ["SPEC_SUFFIX", "expand_spec", "load_generated_cases",
           "load_generated_specs"]
