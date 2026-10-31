# Handoff: continuing the Guarded Agent Ensemble on the professor's machine

Written 2026-10-31 at the end of the AgentDojo phase. Read this file in full, then
`docs/AGENTDOJO_SPIKE.md`, before doing anything. `CLAUDE.md` (the original build spec)
loads automatically.

Repo: https://github.com/coderconnoisseur/guarded-agent-ensemble

## 1. Branches

| Branch | What's on it | PR |
|---|---|---|
| `claude/defense-rev4` | **Work from here.** Defense revisions 4–5, frozen code, `--modules` flag, spike doc §1–23, and all 53 raw AgentDojo result files in `results/agentdojo/` (force-added; `results/` is otherwise gitignored) | #3 (open) |
| `claude/paper-draft-46d6b3` | Paper draft (`paper/`, ACL LaTeX, figure script, claim→source map in `paper/README.md`). Branched from `defense-rev4` before the results commit | none yet |
| `claude/goofy-morse-1a8257` | Block attribution: `scripts/attribute_agentdojo_blocks.py` plus a spike-doc section numbered **§21, which clashes** with defense-rev4's §21. Renumber it to §24 when merging | none yet |
| `claude/nifty-kirch-356319` | AgentDojo's four built-in defenses as `--condition` values (baselines) | #2 (open) |
| `claude/project-readme-comprehensive-47c733` | README and `docs/DEFERRED.md` | #1 (open) |

Merge order when the user asks: #2, then #3, then attribution, then the paper. #2 and #3 conflict in the spike doc.

## 2. What is done (pointers into docs/AGENTDOJO_SPIKE.md)

- **§1–8, setup and bridge.** AgentDojo v1.2.2 (pip `agentdojo==0.1.35`), attack `important_instructions`, suites banking, Slack and travel. Workspace was never run.
- **§9–14, 4B (`qwen3:4b-instruct-2507-q4_K_M`) on banking and Slack.** Condition A vs B, defense revisions 1–3, committed dev/held-out splits.
- **§15, AgentDojo's own defenses** on banking held-out (branch of PR #2).
- **§16, revisions 4–5.** Rev 5 adds value-provenance tracing in the Misalignment Checkpoint. Its utility gain is post hoc.
- **§17–19, travel.** Pre-registered, never tuned on, run on 4B and `qwen2.5:7b`. Security held (4B ASR 15→0, 7B 10→3). Rev 5's utility gain did **not** replicate.
- **§20–21, 7B on banking and Slack** (pre-registered). Banking ASR 27→0, Slack 23→4. Every residual Slack attack on both backbones is `injection_task_3`.
- **§22–23, single-module ablation** (4B, held-out, pre-registered).
  - Planner-only ASR 5/72 and 9/55; Checkpoint-only 0/72 and 7/55; Firewall-only ≈ no defense; full ensemble 0/72 and 8/55.
  - The pre-registered "ensemble beats every module" rule was **not met**.
  - Exploratory: the Planner misses planned calls carrying attacker values, the Checkpoint misses user-sanctioned links.
- **Attribution branch:** which module stopped each attack, from a cache replay. On banking and Slack the Planner blocks first most often; the Firewall blocks no banking attack. Travel is only partly replayable because the Kaggle cache was never downloaded.
- **In-house work (earlier phases):** see `docs/RESULTS_SUMMARY.md`. It includes the AgentHarm run (352 prompts) and the GAI index.

## 3. Rules (non-negotiable; the user insists)

1. **Never fabricate or estimate a number.** Every reported figure comes from a result file or a run. If something isn't measured, say so.
2. **Pre-register before any new experiment.** Append a section to `docs/AGENTDOJO_SPIKE.md` with the question, setup, tests (two-sided Fisher exact on attacked runs), predictions and decision rule, and commit it **before** the first run. Report predictions that fail.
3. **Fail-open runs** caused by a provider error (5xx, rate or daily limit) are removed from the results file and re-run. A failure that repeats deterministically counts as the defense's real behaviour. The only such case so far is travel task 19's truncated plan.
4. **Don't change defense code** for a pre-registered run. It is frozen at `2c292f6`; only the `--modules` CLI flag (`82fa320`) was added since. A defense change means a new revision number and a new pre-registration.
5. **Download and back up results after every run group.** Keep nothing only on one machine. Ask the user before downloads, pushes or anything outward-facing.
6. **Don't line up runs without the user's go-ahead.** Report progress (counts, ASR, fail-open) as runs proceed.
7. **The 4B vs 7B comparison is not a size effect** (different model generations). Never claim to beat the source papers' numbers.

## 4. Experiment plan for this machine (in the order the user discussed)

None of these has been pre-registered yet. Write each pre-registration first.

1. **The 27B as the agent on the full AgentDojo suite** (banking, Slack, travel, workspace): A vs full rev 5, N = 1. Optionally Planner-only and Checkpoint-only.
   - **Size:** 726 runs per condition.
   - **Purpose:** credible headline numbers, and the first workspace result.
   - **Where to run the 27B:** locally (vLLM or Ollama on the GPU), which has no daily cap or API cost, or on OpenRouter (`qwen/qwen3.8-27b`, pin one provider).
   - **Rule:** use the same served model for agent, planner and judge within the experiment.
   - **Check before starting:** whether workspace prompts fit the context window, and the Firewall's false-positive rate on workspace (spike §4.3 measured 28/84 clean outputs flagged offline).
   - **Cost if hosted:** an untested back-of-envelope estimate put it near $40–45 for A plus the full ensemble. Run a 20-run pilot to measure real tokens per run before relying on it.
2. **A size curve within one model family** (e.g. Qwen2.5 3B, 7B, 14B, 32B) on banking and Slack held-out. This separates capability from model generation.
3. **A second attack type** (e.g. AgentDojo's `tool_knowledge`) on banking and Slack held-out, testing whether the "two modules each nearly suffice" finding holds beyond one attack.
4. **Optional: CaMeL** (Debenedetti et al. 2025) on the same tasks, the closest prior design. First spend about an hour checking that its code runs with an open-weight planner.

## 5. Setup notes

- **Python 3.11+.** Run `pip install -r requirements.txt -r requirements-agentdojo.txt`, then `python -m pytest`. Report the count you see; don't assume one.
- **`.env`:** copy `.env.example` to `.env`. It needs `GROQ_API_KEY` for the hosted planner, judge and safety classifiers. The user supplies keys; never paste or echo them.
- **Local backbone:**
  - Set `USE_LOCAL_BACKBONE=1` and `OLLAMA_BASE_URL=http://127.0.0.1:11434/v1`. Use 127.0.0.1, not localhost (IPv6 stall).
  - Start the server with `OLLAMA_CONTEXT_LENGTH` set: 8192 for banking and Slack, 16384 for travel, more for workspace. The client raises an error rather than score a truncated prompt.
  - Set `OLLAMA_REQUEST_TIMEOUT_S=900`.
  - Record the model digest (`ollama list`) in every pre-registration. The 4B is `0edcdef34593`, the 7B `845dbda0ea48`.
- **vLLM:** it serves an OpenAI-compatible API, so point `OLLAMA_BASE_URL` at it. Verify the model name matches an entry in `settings.PROVIDER_CHAIN` (`demos/agentdojo_run.py` exits otherwise).
- **Run command:**

  ```
  python demos/agentdojo_run.py --suite banking --tasks heldout --model <model> --condition B --defense-revision 5 --planner-model qwen/qwen3.8-27b [--modules planner]
  ```

  Results append to `results/agentdojo/<suite>_<cond>_<model>_<attack>_<tags>.json` and resume where they stopped.
- **Groq free tier:** 200k tokens/day on the 27B in a **rolling ~24 h window**, not a midnight reset. A successful test call doesn't prove the budget is back. Only one machine should run defended (Groq) runs at a time. Watch logs for `tokens per day` and stop early. The earlier run scripts had a watcher that did this.
- **Temperature 0 is not bit-reproducible across machines.** Slack no-defense was 37 vs 28 of 55 on laptop vs Kaggle. Re-run the comparators on the same machine when comparing.
- **Long runs on Windows:** launch them so a Claude restart can't kill them (WMI `Invoke-CimMethod Win32_Process Create`). Kill orphaned `llama-server` processes before a run.

## 6. Paper status

Branch `claude/paper-draft-46d6b3`, `paper/main.tex`, which compiles on Overleaf (8 pages). Authors: Nishant Borkar and Jayesh Sharma, NIT Raipur.
- **Revised so far:** each source paper credited once, plain module names, a ~150-word idea-first abstract.
- **Open decision (the user's):** reframe around the professor's theme, "Trustworthy Tool Selection in LLM Agents". **Option A** reframes with the existing results ("constraining which tool and tracing which value"). **Option B** adds a pre-registered tool-description-poisoning experiment.
- The original outline's "risk-adaptive orchestration" was never built. What exists is static risk tags (critical, read-only, returns-untrusted). Don't present an adaptive policy.
