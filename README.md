# Guarded Agent Ensemble

One LLM agent, wrapped in four defense modules each adapted from a different
agent-safety paper, measured before-and-after with a custom composite index
(the **Guarded Agent Index**, GAI) across four test suites.

- **Condition A** — bare backbone, no defenses. The baseline.
- **Condition B** — the same backbone wrapped in all four modules.

The full specification lives in [CLAUDE.md](CLAUDE.md); the interactive
architecture diagram is [docs/architecture.html](docs/architecture.html), with
its companion text in [docs/architecture.md](docs/architecture.md). The six
source papers are in [docs/papers/](docs/papers/).

## Setup

**[docs/REPRODUCING.md](docs/REPRODUCING.md) is the full guide** — what to
install, which key you need, what each command costs, and what will go wrong.
The short version:

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste a free Groq key into it
python -m pytest            # 446 tests, no network, no key needed
```

One free [Groq key](https://console.groq.com/keys) is the only requirement.
The pinned backbone and both auxiliary safety models run on it. OpenRouter and
Gemini keys are optional fallbacks.

`.env` is gitignored, and a value there takes precedence over anything exported
in the shell — so a stale exported key cannot silently shadow a good one.

## Running

```bash
python demos/phase0_demo.py "read the sandbox welcome file and summarize it"
```

Run it twice: the second run serves every LLM call from the disk cache and
spends nothing from the daily request budget.

```bash
python demos/phase1_demo.py
```

Runs the test suite through the unguarded agent and writes
`results/phase1_condition_a_<model>.json`. Narrow it with
`--suite injection`, `--scenario banking` or `--limit 1` to spend fewer
requests.

Test cases are a grid: `scenario` is the tool surface the agent acts on,
`suite` is the threat model, and `enabled_modules` is the defense
configuration. Scenarios are scoped, so an agent on a banking task never sees
the workspace tools - which is both what makes a per-surface number mean
something and what keeps the response cache valid as surfaces are added.

```bash
python demos/frozen_ablation.py --dry-run
```

Estimates the cost of the full benchmark without spending anything. The real
run takes ~4 hours of wall clock — bound by Groq's 2 requests/minute, not by
the daily cap — and resumes where it left off.

## Build status

| Phase | Scope | Demo | State |
|---|---|---|---|
| 0 | Scaffolding, LLM client, tool registry, ReAct loop | `demos/phase0_demo.py` | **built** |
| 1 | Condition A baseline + first test cases | `demos/phase1_demo.py` | **built** |
| 2 | Harm Gate (AgentHarm) | `demos/phase2_demo.py` | **built** |
| 3 | Planner / Tool Dependency Graph (IPIGuard) | `demos/phase3_demo.py` | **built** |
| 4 | Response Firewall + Quarantine (ShieldMCP) | `demos/phase4_demo.py` | **built** |
| 5 | Misalignment Checkpoint (InferAct) | `demos/phase5_demo.py` | **built** |
| 6 | Full A/B, GAI, ablation, report | `demos/phase6_full_eval.py` | not started |

## Layout

```
config/settings.py     model chain, budget/rate limits, GAI weights, paths
src/llm/client.py      OpenRouter wrapper: cache, budget, retry, fallback
src/tools/             vetted registry + three scenario surfaces:
                         workspace (sandboxed files, stubbed web, mock comms)
                         banking   (accounts, payees, transfers)
                         travel    (flight search, bookings)
src/agent/             ReAct loop and prompt templates (Condition A)
src/defense/           the four defense modules (phases 2-5)
src/pipeline/          condition_a / condition_b runners
src/eval/              test suites, runner, GAI scorer, report
demos/                 one runnable demo per phase
sandbox/               the agent's entire filesystem world
```

## Notes on the free tier

The backbone runs on OpenRouter's free tier, which caps requests at 20/minute
and 50/day without purchased credits. Three things follow from that, and they
shape the code rather than just the model choice:

- Every LLM call goes through `LLMClient`, which rate-limits, tracks the daily
  budget on disk, and caches responses so re-running a demo is free.
- Tool calling is a **prompted text protocol** (`Thought:` / `Action: {json}` /
  `Final:`), not OpenRouter's native `tools` parameter, because free models
  report native tool support inconsistently.
- `FREE_MODEL_CHAIN` lists three models from three different providers, so one
  model being retired mid-project does not stop the build.
