You're continuing a multi-phase research project. The repo is at
`D:\Project\MinorProject`, pushed to
https://github.com/coderconnoisseur/guarded-agent-ensemble

Read these in full before doing anything else:
1. `CLAUDE.md` (repo root) — the spec, auto-loaded as project memory.
2. `docs/HANDOFF.md` — start with the **START HERE** box at the top, then
   **§4a** (fresh-worktree setup) and **§12** (current state + your task).
   §1–§11 are a historical build journal; §12 supersedes them on any
   disagreement and several numbers in them have moved.
3. `docs/RESULTS_SUMMARY.md` — every current number, verified. Do not
   re-derive figures from `results/*.json`.

FIRST, because a fresh worktree has no gitignored assets and tests will fail
without them:

    MAIN=D:/Project/MinorProject
    cp $MAIN/.env .
    mkdir -p src/llm/cache results external
    cp -r $MAIN/src/llm/cache/. src/llm/cache/
    cp $MAIN/src/llm/.budget.json src/llm/
    cp -r $MAIN/results/. results/
    cp -r $MAIN/external/. external/

If the main checkout looks stale, check `.claude/worktrees/*/results/` — a
recycled worktree may hold newer data. Then confirm `python -m pytest` gives
**665 passing**. Never state a test count without running pytest first.

## Your task: the AgentDojo spike (HANDOFF §12.4)

My professor asked whether this work has been run on a research-level
benchmark. The honest answer is partially: **AgentHarm yes and properly**
(352 public prompts, held-out split, p=2.1e-13 on the Harm Gate), **AgentDojo
no** — `src/eval/attacks.py` vendors their five attack *templates* into our own
39 hand-written cases, but we have never run inside AgentDojo's environment,
task suite or scoring harness.

That gap is why the project currently cannot compare itself to IPIGuard's or
ShieldMCP's published numbers, and closing it is what would make this
publishable rather than internal-only.

**This is a spike, not an implementation.** I want an informed estimate, not a
half-built adapter. In order:

1. `pip install agentdojo` and **read its current agent-pipeline interface**.
   Do not estimate before reading it — the interface has changed across
   releases and a guess would set the whole plan wrong.
2. Work out what an adapter costs. Our agent is a prompted-JSON ReAct loop
   (`src/agent/loop.py`, CLAUDE.md §5.3) over our own `ToolRegistry`.
   AgentDojo brings its own tools and environments.
3. **The main technical risk**: our defense enforcement lives in registry
   wrappers (`base → Misalignment → PlanEnforcing → Firewall`), deliberately,
   so the agent loop stays byte-identical between Condition A and B. If
   AgentDojo owns tool dispatch, that layering has to be re-expressed. Work out
   whether it survives the move.
4. Report back: what it would take, what it would buy, what could go wrong,
   and whether you recommend doing it. **Then stop and wait** — do not start
   the adapter.

## How I want you to work

* **Measure, don't assume.** Every number in `config/settings.py` has its
  measurement in a comment. Keep that habit. Published limits have been wrong
  twice and my own machine's config once.
* **Read HANDOFF §12.3 before writing code.** Four separate times something in
  this repo was accepted and silently did nothing while producing a plausible
  number. Assume there are more.
* Never state a test count without running `pytest` first.
* Tune and report on different data.
* A defense that silently does nothing is worse than one that crashes.
* Report unflattering results plainly — several already in the repo.
* Flag deviations from `CLAUDE.md`/`docs/architecture.md` rather than silently
  choosing; the diagram is the design source of truth.
* Commit at the end of the work and push to `origin master`.

## Budget and machine notes

* Groq free tier, 1000/day per model, **2 requests/minute** — that rate limit,
  not the daily cap, is what costs wall clock. Check `src/llm/.budget.json`
  before anything large and use `--dry-run` where demos offer it.
* A local Ollama arm exists (`qwen2.5:3b`, ~3.3 s/call, no rate limit). If you
  need it: **`OLLAMA_MODELS` must be `D:\ollama`, not `D:\ollama\models`** —
  the system variable is set one level too deep and a server started without
  the override sees zero models. Start it yourself with the override, and stop
  it when you're done.
* The spike itself should cost no model calls beyond a smoke test.

## Context you may need later (not now)

After the spike, three more things are queued (HANDOFF §12.5): build
`DIV_ASR`, add a third backbone on my professor's 8 GB machine to turn the
capability finding into a curve, and eventually write this up as a paper. Don't
start any of them — I want the spike's answer first because it may reorder the
rest.
