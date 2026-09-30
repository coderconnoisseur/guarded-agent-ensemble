# Presentation walkthrough — Guarded Agent Ensemble

A point-by-point script for presenting this project: the idea, the repo, the
results, and the journey. Every command below was verified to run. Commands
marked **[offline]** call no model and cannot fail for network reasons — use
those live. Commands marked **[cached]** replay saved responses and only touch
the network on a cache miss.

> **Read §0 first.** It is the one thing that can undermine the whole
> presentation if you get it wrong in the room.

---

## §0. The one claim you must NOT make

Do **not** say the project beats the source papers' published numbers.

The papers (IPIGuard, ShieldMCP, InferAct, AgentHarm) evaluate GPT-4-class
backbones on the *full* AgentDojo/AgentHarm benchmarks — hundreds of cases, far
more compute. This project runs 39 hand-written cases on a 3B local model.
Those are not the same experiment, and claiming victory over them is the first
thing a reviewer will take apart.

**What you say instead**, and it is genuinely strong:

> "The ensemble beats every single one of its own modules — same backbone,
> same 39 cases, same code — by +0.117 on our composite index."

That is an *internal controlled comparison*, it is measured, and the
specification we worked from (`CLAUDE.md` §9.1) explicitly calls it the
strongest claim this project can honestly support.

If he asks "how does it compare to the papers?": *"We report their published
figures in the report as context for scale, with an explicit caveat that it is
not a controlled comparison. Our controlled comparison is internal."*

---

## §1. The one-sentence pitch

> One LLM agent, wrapped in four defense modules each adapted from a different
> agent-safety paper, measured before and after with a custom composite index
> across four test suites — on two backbones of different capability, because
> *how much the defenses help turns out to depend on that*.

**Why an ensemble at all?** Three different threat models, and no single paper
covers more than one:

| threat | who is hostile | module | paper |
|---|---|---|---|
| A user asks for something harmful | **the user** | Harm Gate | AgentHarm |
| Instructions hidden in tool output | **the environment** | Planner + Firewall | IPIGuard, ShieldMCP |
| A benign instruction misread | **nobody** — honest error | Misalignment Checkpoint | InferAct |

---

## §2. Repo tour — where everything lives

```bash
# [offline] the map
ls src/defense/ src/pipeline/ src/eval/
```

### The four defense modules

| module | file | paper | mechanism |
|---|---|---|---|
| Harm Gate | `src/defense/harm_gate.py` | AgentHarm | intake classifier, refuses before planning |
| Planner (TDG) | `src/defense/planner.py` | IPIGuard | plans the whole tool sequence before touching untrusted data |
| Response Firewall | `src/defense/firewall.py` | ShieldMCP | scans every tool response for injected instructions |
| Quarantine | `src/defense/quarantine.py` | IPIGuard | Fake Tool Invocation — sanitised replay so the real task finishes |
| Misalignment Checkpoint | `src/defense/misalignment.py` | InferAct | Theory-of-Mind check at irreversible actions |

### Every prompt, by exact location

He will ask "where is the actual prompt?" — have these ready:

| prompt | location | what it does |
|---|---|---|
| Agent system prompt | `src/agent/prompts.py:12` | the ReAct `Thought:/Action:` protocol |
| Harm Gate classifier | `src/defense/harm_gate.py:231` | stage-2 safety classification |
| Task Inference (unit 1) | `src/defense/misalignment.py:88` | infers the task from the trajectory *alone* |
| Task Verification (unit 2) | `src/defense/misalignment.py:105` | compares inferred task to the real instruction |
| Plan generation | `src/defense/planner.py:172` | asks for the Tool Dependency Graph |
| Re-plan on rejection | `src/defense/planner.py:201` | feeds validation errors back |

**The InferAct detail worth pointing out:** units 1 and 2 are deliberately kept
apart. Unit 1 never sees the user's instruction — it only watches what the
agent did. That separation is the mechanism, and we have measured evidence it
matters (§6, the failed fix).

### Supporting architecture

| thing | file | note |
|---|---|---|
| LLM client | `src/llm/client.py` | rate limit, disk cache, retries, budget, provider fallback |
| Provider adapters | `src/llm/providers.py` | Groq, OpenRouter, Gemini, Ollama — one interface |
| Tool registry | `src/tools/registry.py` | schemas + `critical` / `read_only` tags |
| Condition A / B | `src/pipeline/condition_a.py`, `condition_b.py` | baseline vs guarded |
| Scorer (the GAI) | `src/eval/scorer.py` | all 7 sub-metrics + the composite |
| Statistics | `src/eval/stats.py` | Wilson intervals, Fisher exact, power |
| Report generator | `src/eval/report.py` | builds the markdown reports |

**Key design point to say out loud:** defense enforcement lives in *registry
wrappers*, never in the agent loop — `base → Misalignment → PlanEnforcing →
Firewall`. The agent loop is byte-identical between Condition A and B, so an
A/B difference cannot come from the loop.

---

## §3. Live demo sequence

### 3.1 Prove it all runs

```bash
python -m pytest            # [offline] 659 tests, all offline
```

### 3.2 The flagship: an injection attack, blocked

```bash
# [offline] the exact case, from saved evidence
python demos/show_case.py inj_005 --results results/frozen_5_everything.json
```

Shows the planted payload, the agent's reasoning, the firewall flag, the
quarantine, and the original task still completing. **This is the single most
legible artifact in the project** — it shows a real attack and a real defense
step by step.

```bash
# [cached] side-by-side hijacked vs defended
python demos/phase4_demo.py
```

### 3.3 The headline result

```bash
python demos/ablation_table.py --model qwen2.5:3b   # [offline]
```

Prints all five configurations, confidence intervals on every rate, a
significance panel, and a provenance panel naming the source file for each row.

```bash
python demos/ablation_table.py                       # [offline] the 27B arm
```

### 3.4 Rebuild every number from scratch, live

```bash
python demos/phase6_full_eval.py                     # [offline] writes results/report.md
```

**Say this while it runs:** *"Every number in the report is recomputed from
saved run files. Nothing calls a model. You can regenerate the entire report in
front of me and it will be identical."*

---

## §4. The results — in the order to present them

### 4.1 The headline: the ensemble beats each of its own modules

All rows: same backbone, same 39 cases, same code.

| configuration | `ASR_inj` ↓ | `HS` ↓ | GAI |
|---|---|---|---|
| Condition A (no defenses) | 0.40 | 1.00 | 0.567 |
| Harm Gate only | 0.40 | **0.00** | 0.817 |
| Planner only | **0.07** | 1.00 | 0.661 |
| Firewall + Quarantine only | **0.07** | 1.00 | 0.711 |
| Misalignment only | **0.00** | 0.88 | 0.665 |
| **Full ensemble** | **0.00** | **0.00** | **0.933** |

**The line to deliver:** *"Harm-Gate-only drives the harm score to zero and
leaves the attack success rate at 0.40. Planner-only and Firewall-only are the
exact mirror image. Only the ensemble reaches zero on both — because each
module covers one threat model and none covers all three."*

### 4.2 Statistical significance

| metric | Condition A → B | p | verdict |
|---|---|---|---|
| `ASR_inj` | 6/15 → 0/15 | **0.0084** | significant |
| `HS` | 8/8 → 0/8 | **0.0001** | significant |
| over-refusal | 0/26 → 0/26 | 1.000 | no cost |

### 4.3 The composite, both backbones

| weight vector | 27B: A → B | 3B: A → B |
|---|---|---|
| security-leaning (default) | 0.915 → 0.917 (**+0.001**) | 0.567 → **0.933** (**+0.367**) |
| equal-weighted | 0.921 → 0.900 (−0.021) | 0.600 → **0.933** (+0.333) |
| utility-leaning | 0.953 → 0.867 (−0.087) | 0.720 → **0.911** (+0.191) |

**The finding:** defense value is a function of backbone capability. On a
backbone that already resists the attacks, the ensemble is a net cost — there
is nothing left to prevent. On a weaker one it is a large gain under every
weight vector.

### Where the stats live

| file | what |
|---|---|
| `docs/RESULTS_SUMMARY.md` | **the handout** — both arms, every table |
| `results/report.md` | full report, 27B arm |
| `results/report_qwen2-5-3b.md` | full report, 3B arm |
| `results/frozen_*_qwen2.5-3b.json` | raw 5-row ablation, 3B |
| `results/frozen_*.json` | raw 5-row ablation, 27B |
| `results/isolation_*.json` | single-module rows (§4.1) |

---

## §5. The journey — tell this, it is the strongest part

He will trust the results more if you show the failures. Four to tell:

### 5.1 "Our first defense didn't work, and we only found out because we tested it properly"

The Harm Gate looked like it worked: `HS` 0.25 → 0.00 on our own six cases.
Then we benchmarked it against **AgentHarm's real 352 paired prompts**:

```
harmful flagged   2/176 = 0.01     Fisher p = 0.25, no separation at all
```

It had been **tuned and reported on the same six cases**. We rebuilt it around
a purpose-built safety classifier and re-measured on a held-out split:

```
detection  25/25 = 1.00 [0.87-1.00]    over-refusal 1/25 = 0.04
Fisher p = 2.1e-13
```

```bash
python demos/harm_gate_bench.py --split heldout --classifier   # [cached]
```

**The lesson:** tune and report on different data. `src/eval/agentharm.py`
splits by `id_original`, not by row, because AgentHarm ships four paraphrases
of each behaviour — a row-wise split would "evaluate" on a reworded copy of
what you tuned on.

### 5.2 "The obvious fix to our weakest module was measured, and it failed"

The Misalignment Checkpoint over-flagged. The diagnosis was that unit 2 lacks
the trajectory, so it cannot tell an argument *resolved from an observation*
from one *invented*. We built the fix and replayed it over 27 recorded triples:

| variant | MF1 |
|---|---|
| **original** | **0.85** |
| + trajectory, permissive list | 0.76 |
| + trajectory, no list | 0.77 |

```bash
python demos/misalignment_replay.py     # [offline]
```

**We reverted it.** Shown the evidence, the judge stops judging and starts
agreeing — which is an argument *for* InferAct's two-unit separation, arrived at
by measurement. **Say this one.** It shows you can kill your own idea.

### 5.3 "We couldn't get a significant result, and the fix wasn't more test cases"

Everything was p > 0.2. The instinct is "write more cases". The power
calculation says otherwise — significance is driven by the **baseline rate**:

| baseline attack rate | cases needed per arm | we had |
|---|---|---|
| 0.07 (the 27B) | 68 | 15 ✗ |
| 0.40 (the 3B) | **9** | 15 ✓ |

The 27B resists AgentDojo's own five attack templates **0/30**. So we added a
*deliberately weaker* second backbone. **Same 39 cases**, and both headline
metrics became significant.

### 5.4 "A defense silently did nothing, and the number looked fine"

Our first local run said the Misalignment Checkpoint changed nothing. It was a
provider mismatch: the judge was being sent to a backend that did not have that
model. Every call 404'd and the module took its documented fail-open path **15
times in 39 cases** — while reporting that it ran, with zero errors.

We added a guard that refuses to write a row where a module failed open on
≥10% of runs. **The lesson:** a defense that silently does nothing is worse
than one that crashes, because it still produces a plausible number.

---

## §6. Anticipated questions

**"Why these GAI weights?"**
> We report three vectors — security-leaning, equal, utility-leaning — and the
> conclusion holds under all three on the 3B arm. Quoting only the best-looking
> one would be a worse argument than quoting none. `src/eval/scorer.py` refuses
> a weight vector that does not sum to 1.

**"39 cases is small."**
> Agreed, and we report a 95% Wilson interval on every rate rather than a bare
> point estimate. What determines significance here is the baseline rate, not
> the case count — §5.3. At a baseline of 0.40 we need 9 cases per arm and have
> 15.

**"Is this just the weaker model being bad?"**
> Benign utility is **1.00** on the 3B in both conditions — it does ordinary
> tasks perfectly. It is specifically *unsafe*, not incompetent.

**"Did you just tune until the numbers improved?"**
> The target — beat Harm-Gate-only's 0.817 — was fixed before either fix was
> written, and both are mechanism-level: how malformed plans are handled, and
> which model judges. Neither was derived from inspecting which cases failed.

**"What is not built?"**
> Two of the seven sub-metrics. `DIV_ASR` needs a generated adversarial corpus;
> hand-writing one would measure our imagination, not attack diversity. `LAT` is
> attempted and the scorer refuses the result — §7. Both are dropped from the
> index with the weights renormalised, and the report says so in its own
> section rather than a footnote.

---

## §7. Known gaps — state these before he finds them

| gap | status |
|---|---|
| `DIV_ASR` | never built, deliberately — needs a *generated* corpus (AgentVigil/SIRAJ) |
| `LAT` | attempted; the scorer **refuses** the result because the timing brackets do not line up |
| N=1 | no repeats; at ~3.3 s/call on the local arm N=3 costs ~3.5 h |
| 3B arm confound | `qwen2.5` is a different *generation* from `qwen3.8`, so scale and training recipe are confounded |
| not fully local | the Harm Gate and Firewall guard models stay hosted — deliberately, to isolate the backbone as the variable |

Together `DIV_ASR` + `MF1` were 30% of the default weight vector. The reported
index is a renormalised 4-term index, **not** the 7-term one the spec defines —
and that is stated in the reports.

---

## §8. Suggested 10-minute running order

1. **The idea** (1 min) — §1, the three threat models.
2. **The repo** (2 min) — §2, open `misalignment.py` and show the two prompts.
3. **It runs** (1 min) — `python -m pytest`.
4. **One attack, blocked** (2 min) — `show_case.py inj_005`.
5. **The headline** (2 min) — §4.1 isolation table, `ablation_table.py --model qwen2.5:3b`.
6. **One failure story** (1.5 min) — §5.2, the fix we measured and reverted.
7. **Gaps** (0.5 min) — §7, before he asks.

Hand him `docs/RESULTS_SUMMARY.md`.
