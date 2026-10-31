# Bootstrap: set up and continue the Guarded Agent Ensemble on this machine

You are Claude Code on a machine that has never seen this project. This one file is everything
you need. Work through the steps in order and report briefly after each. The user is
**Nishant Borkar** (B.Tech CSE, NIT Raipur). This is his research project, and he works with
**Jayesh Sharma**. Keep replies short and plain.

Ask the user before anything that installs system software, downloads large models, spends
money, pushes to GitHub or starts a long run. Everything else in steps 1–6 you can just do.

## Step 1: check the machine

Report:
- OS
- `git --version`
- Python version (3.11+ needed)
- `nvidia-smi` (GPU model and VRAM, or none)
- free disk space
- whether `ollama` or `vllm` is installed

Don't install anything yet.

## Step 2: get the code

In the current folder, unless the user names another:

```
git clone https://github.com/coderconnoisseur/guarded-agent-ensemble.git
cd guarded-agent-ensemble
git checkout claude/defense-rev4
```

`claude/defense-rev4` is the working branch: code, docs, and all 53 raw AgentDojo result files
under `results/agentdojo/`. From now on, work inside this folder. Its `CLAUDE.md` is the
original project spec.

## Step 3: Python environment

Create a virtual environment in the repo (`python -m venv .venv`) and activate it. Then:

```
pip install -r requirements.txt -r requirements-agentdojo.txt
```

## Step 4: API key

Copy `.env.example` to `.env`. Ask the user to open `.env` and paste their own
`GROQ_API_KEY` into it. They must type it into the file themselves, never into this chat;
never print or echo the key. Wait until they say it's done. Then check the key works with one
tiny request to `qwen/qwen3.8-27b` that prints only the HTTP status.

## Step 5: tests

Run `python -m pytest -q` and report the exact pass/fail count you see. Never state a test
count you didn't just run.

## Step 6: recreate the project memory

Write these notes into **your own memory directory for this project**; your system prompt
gives its path. Create one file per note, plus a `MEMORY.md` index line for each. They carry
lessons learned on the old machines.

**Note `groq-daily-token-limit` (type: project).**
- Groq caps `qwen/qwen3.8-27b` at 200,000 tokens per day per key, in a **rolling ~24 h window**, not a midnight reset. A successful test call doesn't prove the budget is back.
- The 27B is the planner and judge for defended (Condition B) runs. When the cap is hit, B runs fail open and are recorded with `failed_open` flags. Those rows must be removed and re-run.
- Only one machine runs Condition B at a time. Condition A (no defense) uses no Groq tokens and can run anytime.
- After every B run, check rows for `failed_open` before reporting.

**Note `download-results-every-group` (type: feedback).** On 2026-10-27 an idle Kaggle session shut down and wiped 11 result files that existed only there.
- Results must never live on only one machine. After every run group, back results up: ask the user whether to commit them to a branch, copy them to a drive, or both.
- Ask for standing permission once, then do it every time.

**Note `long-runs-survive-restarts` (type: feedback).**
- Launch multi-hour jobs (model server plus AgentDojo runner) so a Claude restart can't kill them. On Windows use `Invoke-CimMethod Win32_Process Create`; on Linux use `nohup`/`setsid` or tmux. Always log to a file.
- Use `127.0.0.1`, never `localhost`, for the local model server (IPv6 stall).
- Stopping `ollama serve` does not stop its `llama-server` children. Kill the whole process tree and check `nvidia-smi` before a run; orphans have caused CUDA out-of-memory errors.
- Runs are resumable: relaunching loses at most one in-flight run.

**Note `user-research-rules` (type: feedback).** Nishant's non-negotiable rules:
- never fabricate or estimate a number;
- pre-register every experiment before running it;
- report failed predictions;
- don't line up runs without his go-ahead;
- keep claims honest (no inflation; never claim to beat the source papers' numbers).

## Step 7: catch up on the project

Read `docs/NEXT_SESSION_PROMPT.md` (the handoff) in full, then `docs/AGENTDOJO_SPIKE.md`
(sections 1–23, the experimental record). Summarise in 5–8 lines what's done and what the
handoff says is next. Don't re-derive numbers from scratch; the spike doc and the result files
are the record.

## Step 8: plan this machine's work, then stop and ask

From the GPU you found in step 1, propose how to serve the models. **Get the user's OK before
installing anything or downloading models.**

- **GPU with ≥40 GB VRAM (e.g. an A100):** serve the 27B locally. vLLM needs Linux or WSL; Ollama also works. That means no API cost and no daily cap. The same local 27B should be agent, planner and judge within an experiment.
  - Point `OLLAMA_BASE_URL` at the server's OpenAI-compatible `/v1` endpoint and set `USE_LOCAL_BACKBONE=1`.
  - Make sure the served model name is in `settings.PROVIDER_CHAIN`; `demos/agentdojo_run.py` exits otherwise. Read `config/settings.py` before changing anything.
- **Smaller GPU:** use Ollama for the 4B/7B family (`qwen3:4b-instruct-2507-q4_K_M`, `qwen2.5:7b`). For a hosted 27B, use Groq (free, daily cap) or OpenRouter (paid, which needs the user's card).
- **Always:**
  - Start the server with an explicit context length: 8192 for banking and Slack, 16384 for travel, and more for workspace (measure first). The client refuses truncated prompts.
  - Set `OLLAMA_REQUEST_TIMEOUT_S=900`.
  - Record the model digest or weights hash for the pre-registration.

Then propose the **first pre-registration**: the 27B as the agent on the full AgentDojo suite
(banking, Slack, travel, workspace), Condition A vs full ensemble at defense revision 5, N = 1.
Planner-only and Checkpoint-only are optional.
- **Write it in the style of the existing pre-registrations** (spike doc §17, §20, §22): question, setup, tests (two-sided Fisher exact on attacked runs), predictions, decision rule.
- **Use the next free section number.** The attribution branch already uses a §21 that clashes, so check.
- **First check workspace feasibility:** prompt sizes against the context window, and how often the Firewall flags clean workspace outputs (spike §4.3 found 28/84 offline).
- **Before the full run,** run a small pilot (≈20 runs) to measure time and tokens per run.

**Stop here** and show the user the serving plan and the pre-registration draft. Do nothing
more until they approve. After approval: commit the pre-registration, run in groups, report
progress (runs done, attacks through, fail-open count) as it goes, and back up results after
every group.

## Rules (from the handoff; repeated so you can't miss them)

1. Every reported number comes from a real run or result file. If it isn't measured, say so.
2. Pre-register before any run, and commit the pre-registration first.
3. Remove and re-run fail-open rows caused by provider errors. A deterministic, repeatable failure counts.
4. Don't edit defense code for a pre-registered run. It is frozen at commit `2c292f6`, plus the `--modules` CLI flag. A change means a new revision and a new pre-registration.
5. Temperature 0 is not bit-reproducible across machines. Re-run comparators on this machine instead of comparing to laptop or Kaggle numbers.
6. The 4B vs 7B comparison is a different-generation comparison, not a size effect.
7. Ask before pushing, merging PRs, spending money or downloading big files.

## What lives only on Nishant's laptop (not needed here)

The Groq LLM disk cache, the old results backups (all raw results are also in the repo now),
and the Overleaf login for the paper. The paper source is on branch
`claude/paper-draft-46d6b3` (`paper/main.tex`) if he asks you to work on it.
