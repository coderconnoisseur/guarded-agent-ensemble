"""Replay, from the LLM cache only, every held-out run where Condition B rev 1
lost utility that Condition A had, and print what each defense module decided.

Run from the AgentDojo worktree root. Network is pointed at dead endpoints, so a
cache miss errors instead of making a live call.
    python <this> banking
"""
import json, os, sys
os.environ.update(USE_LOCAL_BACKBONE="1", OLLAMA_BASE_URL="http://127.0.0.1:9/v1",
                  HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9",
                  NO_PROXY="", HF_HUB_OFFLINE="1")
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "demos")]

import agentdojo_run as R  # noqa: E402
from config import settings  # noqa: E402
from src.agentdojo_bridge import GuardedPipeline  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.pipeline.condition_b import ConditionB  # noqa: E402

suite_name = sys.argv[1]
model = "qwen3:4b-instruct-2507-q4_K_M"
d = "results/agentdojo/"
held = {"banking": "heldout_rev1", "slack": "rev1"}[suite_name]
B = json.load(open(f"{d}{suite_name}_B_qwen3-4b-instruct-2507-q4_K_M_important_instructions_{held}.json"))
A = {(r["user_task"], r["injection_task"]): r for r in
     json.load(open(f"{d}{suite_name}_A_qwen3-4b-instruct-2507-q4_K_M_important_instructions.json"))}
skip = {"banking": {"user_task_8", "user_task_10"}}.get(suite_name, set())
lost = [(r["user_task"], r["injection_task"]) for r in B
        if r["injection_task"] and not r["utility"] and A[(r["user_task"], r["injection_task"])]["utility"]
        and r["user_task"] not in skip]
print(f"{len(lost)} lost runs")

settings.DEFENSE_REVISION = 1
suite = R.get_suites(R.BENCHMARK_VERSION)[suite_name]


def short(x, n=400):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


with LLMClient() as client:
    pipe = GuardedPipeline(lambda reg: ConditionB(client, registry=reg, model=model,
                                                  max_steps=R.MAX_STEPS), "replay")
    attack = R.load_attack("important_instructions", suite, pipe)
    for ut_id, it_id in lost:
        ut, it = suite.user_tasks[ut_id], suite.injection_tasks[it_id]
        utility, security = suite.run_task_with_pipeline(pipe, ut, it, attack.attack(ut, it))
        res = pipe.last_result
        print(f"\n===== {ut_id} x {it_id}  utility={utility} (recorded False)  "
              f"calls={res.num_llm_calls} hits={res.num_cache_hits} stop={res.stop_reason} err={res.error}")
        print("TASK:", ut.PROMPT)
        print("TOOLS:", [(n, a) for n, a in res.tool_calls])
        for name in ("harm_gate_verdict", "plan_enforcement", "firewall_verdicts",
                     "quarantine_events", "misalignment_verdicts"):
            print(f"{name}:", short(getattr(res, name), 900))
        print("FINAL:", res.final_answer[:300])
