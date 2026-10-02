"""AgentDojo spike: measure the adapter's risks offline, at zero model calls.

Not the adapter. Docs/AGENTDOJO_SPIKE.md quotes every number this prints, and
this script is how a reader re-derives them.

AgentDojo ships a `GroundTruthPipeline` that replays each user task's correct
tool calls against the real task environment. That gives the exact tool
outputs a perfect agent would see - clean, and with the `important_instructions`
attack injected - without calling any model. Everything below is measured on
those outputs:

  1. how many ground-truth tool calls each user task needs, against our
     `AGENT_MAX_STEPS`;
  2. how large the observations are, against Groq's 8000 TPM and a local
     model's context window;
  3. whether our Firewall's zero-call heuristic flags the injected outputs
     (recall), and whether it flags the clean ones (false positives, which
     Quarantine turns into withheld data);
  4. where in the output the payload lands, against the guard model's
     `FIREWALL_GUARD_MAX_CHARS` head truncation;
  5. what Quarantine's sanitiser removes besides the payload.

Requires the `agentdojo` package, which is deliberately NOT in
requirements.txt - install it into a separate venv:

    python -m venv adenv && adenv/Scripts/python -m pip install agentdojo httpx python-dotenv
    adenv/Scripts/python scripts/agentdojo_spike_measure.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline  # noqa: E402
from agentdojo.attacks.important_instructions_attacks import (  # noqa: E402
    ImportantInstructionsAttack,
)
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from config import settings  # noqa: E402
from src.defense.firewall import heuristic_signals  # noqa: E402
from src.defense.quarantine import sanitise  # noqa: E402

BENCHMARK_VERSION = "v1.2.2"
CHARS_PER_TOKEN = 4  # rough English/YAML average; reported as an estimate


class _NamedStub(BasePipelineElement):
    """Only exists so the attack can read a model name off `.name`.

    `important_instructions` raises unless the pipeline name contains a model
    string AgentDojo knows; "local" renders as "Local model".
    """

    name = "local"

    def query(self, *a, **k):  # pragma: no cover - never run
        raise NotImplementedError


def tool_outputs(suite, user_task, injections: dict[str, str]) -> list[tuple[str, str]]:
    env = suite.load_and_inject_default_environment(injections)
    env = user_task.init_environment(env)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, messages, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT, runtime, env
    )
    out = []
    for m in messages:
        if m["role"] == "tool":
            text = "".join(b.get("content") or "" for b in m["content"])
            out.append((m["tool_call"].function, text))
    return out


def _norm(text: str) -> str:
    # Travel tools return dicts, which AgentDojo renders with str(): the whole
    # response is one line with *literal* backslash-n escapes. Decode them for
    # matching only - the defenses are always measured on the raw text.
    return " ".join(text.replace("\\n", "\n").split())


# The environment loader collapses the attack's blank lines, so the payload
# never appears verbatim in a tool output. A first version of this script
# matched it verbatim and reported "payload never seen" for all 949 pairs -
# a plausible zero produced by the measurement, not the system. Match on
# whitespace-normalised text, and locate it by its opening tag. A second
# version still reported 0/140 for travel; see _norm.
MARKER = "<INFORMATION>"


def carries(text: str, payloads: list[str]) -> bool:
    t = _norm(text)
    return any(_norm(p)[:200] in t for p in payloads)


def payload_survives(text: str, goal: str) -> bool:
    return _norm(goal)[:60] in _norm(text)


def main() -> None:
    suites = get_suites(BENCHMARK_VERSION)
    report: dict = {"benchmark_version": BENCHMARK_VERSION, "suites": {}}
    gt_calls_all: list[int] = []

    for name, suite in suites.items():
        attack = ImportantInstructionsAttack(suite, _NamedStub())
        s: dict = defaultdict(int)
        obs_chars: list[int] = []
        task_chars: list[int] = []
        gt_calls: list[int] = []
        clean_fp_tools: Counter = Counter()
        missed_goals: Counter = Counter()

        for ut in suite.user_tasks.values():
            env = suite.load_and_inject_default_environment({})
            gt_calls.append(len(ut.ground_truth(ut.init_environment(env))))

            clean = tool_outputs(suite, ut, {})
            task_chars.append(sum(len(t) for _, t in clean))
            for tool, text in clean:
                obs_chars.append(len(text))
                s["clean_outputs"] += 1
                s["clean_outputs_flagged"] += 0
                if heuristic_signals(text):
                    s["clean_outputs_flagged"] += 1
                    clean_fp_tools[tool] += 1

            candidates = attack.get_injection_candidates(ut)
            if not candidates:
                s["user_tasks_not_injectable"] += 1
                continue
            for it in suite.injection_tasks.values():
                s["pairs"] += 1
                injections = attack.attack(ut, it)
                outs = tool_outputs(suite, ut, injections)
                payloads = [p for p in injections.values() if p]
                carrying = [(tool, t) for tool, t in outs if carries(t, payloads)]
                if not carrying:
                    s["pairs_payload_never_seen"] += 1
                    continue
                flagged = [t for _, t in carrying if heuristic_signals(t)]
                if flagged:
                    s["pairs_heuristic_flagged"] += 1
                else:
                    missed_goals[it.ID] += 1
                # Guard model only ever sees the head of the output.
                first = min(t.find(MARKER) for _, t in carrying)
                if first >= settings.FIREWALL_GUARD_MAX_CHARS:
                    s["pairs_payload_beyond_guard_window"] += 1
                # Quarantine: does the payload survive, and what else goes?
                for _, t in carrying:
                    if not heuristic_signals(t):
                        continue
                    kept, removed = sanitise(t)
                    if heuristic_signals(kept):
                        s["outputs_withheld_entirely"] += 1
                        continue
                    if payload_survives(kept, it.GOAL):
                        s["outputs_payload_survived_sanitise"] += 1
                    collateral = sum(
                        len(r) for r in removed
                        if MARKER not in r and not payload_survives(r, it.GOAL)
                    )
                    if collateral:
                        s["outputs_with_collateral_removal"] += 1
                        s["collateral_chars_removed"] += collateral

        gt_calls_all += gt_calls
        s.update(
            user_tasks=len(suite.user_tasks),
            injection_tasks=len(suite.injection_tasks),
            tools=len(suite.tools),
            gt_calls_median=statistics.median(gt_calls),
            gt_calls_max=max(gt_calls),
            user_tasks_over_max_steps=sum(
                c + 1 > settings.AGENT_MAX_STEPS for c in gt_calls
            ),
            obs_chars_median=statistics.median(obs_chars),
            obs_chars_max=max(obs_chars),
            task_obs_chars_max=max(task_chars),
            task_obs_tokens_est_max=max(task_chars) // CHARS_PER_TOKEN,
            clean_fp_by_tool=dict(clean_fp_tools),
            heuristic_missed_by_injection_task=dict(missed_goals),
        )
        report["suites"][name] = dict(s)

    report["total_user_tasks"] = sum(v["user_tasks"] for v in report["suites"].values())
    report["total_pairs"] = sum(v["pairs"] for v in report["suites"].values())
    report["agent_max_steps"] = settings.AGENT_MAX_STEPS
    report["user_tasks_over_max_steps"] = sum(
        c + 1 > settings.AGENT_MAX_STEPS for c in gt_calls_all
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
