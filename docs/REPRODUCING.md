# Running this yourself

Everything here has been run on a Windows 10 laptop with no GPU, on free API
tiers, with no paid credit. You need one free API key.

---

## 1. What you need

- **Python 3.11+**
- **One free Groq API key** — <https://console.groq.com/keys>. No card, no
  waitlist. This is the only key required; the pinned backbone and both
  auxiliary safety models run on it.
- ~20 minutes for the demos. **Several hours** for the full benchmark — see
  §6, and read it before you start one.

OpenRouter and Gemini keys are optional fallbacks. Leave them blank.

---

## 2. Setup

```bash
git clone https://github.com/coderconnoisseur/guarded-agent-ensemble.git
cd guarded-agent-ensemble
pip install -r requirements.txt
cp .env.example .env
```

Then paste your Groq key into `.env`:

```
GROQ_API_KEY=gsk_...
```

`.env` is gitignored, and a value there beats anything exported in your
shell — a stale exported key otherwise silently wins and the failure looks
like a bad key.

---

## 3. Verify it works without spending anything

```bash
python -m pytest
```

**446 tests, zero network calls, no API key needed.** HTTP is stubbed with
`httpx.MockTransport` throughout. If this passes, the install is good.

This is also the fastest way to see what the project claims: the tests are
written as statements about behaviour, and several encode bugs that were
found the hard way (`tests/test_misalignment.py`,
`tests/test_scenarios.py`).

---

## 4. Your first live call

```bash
python demos/phase0_demo.py "read the sandbox welcome file and summarize it"
```

Spends ~2 requests. Run it **twice**: the second run serves every call from
the disk cache (`src/llm/cache/`) and spends nothing. Watch for
`from_cache: true`.

If it fails, the two usual causes are an empty `GROQ_API_KEY` and a model that
Groq has retired. `config/settings.py` documents how each model was verified
live before being pinned.

---

## 5. The demos, cheapest first

Each is one command and prints its own explanation. Costs are for a cold
cache; a second run of anything is free.

| Command | Shows | ~Requests |
|---|---|---|
| `python demos/phase0_demo.py "<task>"` | the bare ReAct loop calling a real tool | 2 |
| `python demos/phase2_demo.py --ablation` | Harm Gate: `HS` with and without it | ~25 |
| `python demos/phase3_demo.py --case inj_005` | the Tool Dependency Graph blocking an off-plan call | ~15 |
| `python demos/phase4_demo.py` | **flagship** — an injection hijacking the bare agent, then quarantined and completed anyway | ~20 |
| `python demos/phase5_demo.py` | the ToM checkpoint pausing on an overreaching delete | ~15 |
| `python demos/phase5_demo.py --mf1` | misalignment detection quality across both classes | ~50 |
| `python demos/show_case.py inj_005` | any saved case, rendered legibly | 0 |
| `python demos/ablation_table.py` | the cumulative ablation, with its evidence | 0 |

**Start with `phase4_demo.py`.** It prints both transcripts side by side:
Condition A follows the injected instruction, Condition B flags it,
quarantines it, and still finishes the user's real task.

Narrow any run to spend less:

```bash
python demos/phase1_demo.py --scenario banking     # one column of the grid
python demos/phase1_demo.py --suite injection --limit 3
```

---

## 6. The full benchmark

```bash
python demos/frozen_ablation.py --dry-run   # cost estimate, spends nothing
python demos/frozen_ablation.py             # the real thing
```

This runs all five defense configurations over one frozen 39-case set on one
pinned backbone — the only version of the cumulative ablation that can
honestly be read as a trend.

**Budget honestly before starting.** Measured, not estimated:

- ~500 fresh requests, well inside Groq's 1000/day.
- **~4 hours of wall clock.** The binding constraint is not the daily cap —
  it is Groq's output-tokens-per-minute ceiling of 1000, which with a 400-token
  reservation per call works out at **2 requests per minute**.

So it will be interrupted, and it is built for that: each row is written to
`results/` as it completes and a row already on disk is skipped on the next
run. Just run it again. `--force` re-runs everything.

When all five rows exist:

```bash
python demos/ablation_table.py
```

It prints the five rows, the file each row came from, and — only if all five
cover the same case set on the same backbone — accepts the trend. If they do
not, it **refuses to draw one** and tells you why. That refusal is deliberate:
four numbers going down looks like evidence whether or not the denominators
match.

---

## 7. Reading the output

- `results/` (gitignored) holds one JSON per run: the full transcript, every
  defense module's verdict, the post-run world state, and the graded outcome.
  Any number in the report is recomputable from these without re-running a
  model.
- Grading is **mechanical** — `src/eval/grading.py` evaluates predicates over
  the transcript and world state. No LLM judges a test case. The one LLM judge
  in the project is the Misalignment Checkpoint, where the judgement *is* the
  mechanism being studied (InferAct), not a grading shortcut.
- `python demos/show_case.py <case_id>` renders any saved case: prompt,
  transcript, each check and why it passed or failed.

---

## 8. Limits you will hit

| Symptom | Cause | What to do |
|---|---|---|
| Everything is slow, ~2 calls/minute | Groq's 1000 OTPM ceiling, not a bug | expected; use cached demos, or leave long runs in the background |
| `BudgetExceededError` | 1000 requests/day per model | wait for the day rollover, or re-run — completed rows are on disk |
| A demo spends nothing and finishes instantly | the disk cache | that is the intended behaviour; delete `src/llm/cache/` to force fresh calls |
| `Request too large ... OTPM: Limit 1000` | `DEFAULT_MAX_TOKENS` raised above ~500 | leave it at 400; the value is measured from our own p95 completion length |

---

## 9. Swapping the backbone

One line in `.env`:

```
BACKBONE_MODEL=llama-3.3-70b-versatile
BACKBONE_PROVIDER=groq
```

**This invalidates every prior measurement.** Results are not comparable
across backbones, `results/` filenames carry the model to stop one silently
overwriting another, and `ablation_table.py` ignores snapshots from any model
but the pinned one. Re-run the baseline if you change it.

`python demos/compare_backbones.py` reports per-backbone metrics and where
their failures overlap.

---

## 10. What is not in the repo

- **The six source papers.** `docs/papers/*.pdf` is gitignored — this repo is
  public and those are other people's copyrighted work. `docs/papers/README.md`
  says where to get each one. You do not need them to run anything; `CLAUDE.md`
  §3 contains the condensed mechanism reference the code was written against.
- **`src/eval/testsuites/diversity/`** is deliberately empty. `DIV_ASR` is
  meant to score a *generated* corpus from an AgentVigil-style mutation loop;
  hand-writing cases would misrepresent what that number measures. See the
  README in that folder.
- **`results/` and `src/llm/cache/`** are gitignored, so your first runs will
  be cold and will cost real requests.
