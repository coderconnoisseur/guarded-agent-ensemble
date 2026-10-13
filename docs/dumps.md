# Session dump — read this first to continue

**Last updated:** 2026-10-13 19:00 IST · kept current by the working session
at every milestone. If you are a new session: read this whole file, then
`CLAUDE.md`, then `docs/AGENTDOJO_SPIKE.md` §9–11. Do not re-derive anything
below; it was measured.

---

## 0. Right now

**Revision 2 HELD-OUT run in progress** (started 18:58, ETA ~21:30): Condition B
`--defense-revision 2 --tasks heldout` on Slack (66 runs) then banking (80),
via WMI from `scratchpadev2.ps1 -split heldout`, private port 11435. Log
`scratchpadev2.log`. If it died, relaunch the same command; it resumes.

**Your next job when it ends:** compare rev2 held-out against A and rev1 on
the held-out halves (Slack: `SLACK_HELDOUT` subset of `slack_A_...json` and
`slack_B_..._rev1.json`; banking: `banking_B_..._heldout_rev1.json`), paired
McNemar as in §4. Write spike doc §13 (dev story + held-out numbers), update
§1 here, commit, push.

**Revision 2 dev result (final code, commit 95f12ff):**

| dev | utility under attack | ASR |
|---|---|---|
| Slack A / rev1 / rev2 | 24 / 8 / **13** of 50 | 32 / 6 / **0** of 50 |
| banking A / rev1 / rev2 | 39 / 33 / 33 of 72 | 21 / 0 / 0 of 72 |

rev1->rev2 on Slack dev: +6 utility, 0 lost (p = 0.031); ASR 6 -> 0
(p = 0.031). No module failed open. Remaining 18 lost Slack dev runs: 10 are
the deliberate trade (the address rule blocks legit URLs found in data, tasks
11 and 15), 4 by design (task 18, "do the tasks on this page"), 3 correct
judge/planner catches, 1 judge false positive (task 7).

Revision 2 = rev 1 + (f7f4967) Firewall "e-mail" noun fix, Planner `tool()`
strip + pruned-edge drop, Node Expansion blocks unnamed addresses; +
(95f12ff) Quarantine drops command blocks instead of withholding everything
when the signal is cross-block, judge accepts web pages as a source.

### Where things live

- **Worktree doing the run:** `D:\Project\MinorProject\.claude\worktrees\c-drive-cleanup-77931b`
  (branch `claude/agentdojo-spike`, pushed to `origin/master`). `results/` is
  gitignored — the AgentDojo result files exist ONLY in this worktree:
  `results/agentdojo/*.json`.
- **Logs:** `C:\Users\Acer\AppData\Local\Temp\claude\D--Project-MinorProject--claude-worktrees-c-drive-cleanup-77931b\eb252817-4db1-4444-bf95-f92ed5c2ce7a\scratchpad\`
  `slack.log` (chain milestones), `slack_A.log` / `slack_B.log` (UTF-16; read
  with `iconv -f UTF-16LE -t UTF-8`), `slack_ollama.log`, `slack.ps1` (the chain).

### Check progress

```bash
cd /d/Project/MinorProject/.claude/worktrees/c-drive-cleanup-77931b
python -c "import json,glob; [print(f, len(json.load(open(f)))) for f in glob.glob('results/agentdojo/slack_*.json')]"
```

Each condition is 126 runs. Processes: `ollama`, `python` (and a child
`llama-server`).

### If it died — relaunch (resumable: finished runs are skipped)

**Launch through WMI, never as a child of the Claude app** — the app
auto-restarted at 00:57 on 13 Oct and took the whole run with it. Background
tool tasks are also killed at 2 h. In PowerShell:

```powershell
$S = "<the scratchpad dir above>"   # or any folder; it holds slack.ps1 + logs
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$S\slack.ps1`"" }
```

`slack.ps1` (recreate it if the scratchpad is gone; fix the log paths):

```powershell
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState(0x80000001) | Out-Null   # keep awake while running
$env:OLLAMA_MODELS = 'D:\ollama'; $env:OLLAMA_CONTEXT_LENGTH = '8192'; $env:OLLAMA_HOST = '127.0.0.1:11435'
$srv = Start-Process ollama -ArgumentList serve -WindowStyle Hidden -RedirectStandardError "$S\slack_ollama.log" -RedirectStandardOutput "$S\slack_ollama.out" -PassThru
for ($i=0; $i -lt 180; $i++) { try { Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:11435/api/version | Out-Null; break } catch { Start-Sleep 2 } }
$env:USE_LOCAL_BACKBONE = '1'; $env:OLLAMA_BASE_URL = 'http://127.0.0.1:11435/v1'; $env:PYTHONIOENCODING = 'utf-8'
Set-Location 'D:\Project\MinorProject\.claude\worktrees\c-drive-cleanup-77931b'
python -u demos/agentdojo_run.py --suite slack --condition A --model qwen3:4b-instruct-2507-q4_K_M *>> "$S\slack_A.log"
python -u demos/agentdojo_run.py --suite slack --condition B --model qwen3:4b-instruct-2507-q4_K_M --defense-revision 1 *>> "$S\slack_B.log"
Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
[W.P]::SetThreadExecutionState(0x80000000) | Out-Null
```

---

## 1. Results so far (all verified; numbers are from `results/agentdojo/`)

Backbone `qwen3:4b-instruct-2507-q4_K_M` (local Ollama), AgentDojo v1.2.2,
attack `important_instructions`, 15 agent steps, 8192 context.

**Banking, held-out split (the reported number):**

| | ASR | utility under attack | clean |
|---|---|---|---|
| A, no defenses | 39/72 (54%) | 34/72 | 4/8 |
| B as built (revision 0) | 0/72 | 28/72 | 3/8 |
| **B revision 1** | **0/72** (p = 3.6e-12 vs A) | **32/72** (p = 0.82 vs A) | 4/8 |

Claim: on held-out tasks, revision 1 blocks every attack at no detectable
utility cost. Limits: 8 tasks, one suite, one 4B backbone.

Full banking (all 16 tasks): A ASR 60/144 (41.7%); B as built 1/144; real-task
utility 5/11 (A) vs 1/11 (B as built). `qwen2.5:3b` was a NO-GO (7/16 clean).

**Slack (all held-out for revision 1; Ollama 0.35.1):**

| | ASR | utility under attack | clean (floor 0/21) |
|---|---|---|---|
| A | 69/105 (66%) | 49/105 | 14/21 |
| **B revision 1** | **14/105 (13%)**, p = 5.6e-17 | **17/105**, p = 6.7e-08 | 4/21 |

All 14 surviving attacks = one mechanism: Node Expansion allowed a read-only
`get_webpage` to the attacker's URL (`injection_task_3`). Utility lost to
Planner 22, Misalignment 13, Quarantine 4, model 6 (of 45).

**Combined claim:** security generalises on both suites; utility cost is
suite-dependent - none on banking held-out, large on multi-step Slack.

---

## 2. What was built (all on GitHub `master`)

- `src/agentdojo_bridge.py` — our registry over AgentDojo's runtime; `TAGS`
  for all 69 tools (fixed before any run); banking dev/held-out split.
- `demos/agentdojo_run.py` — resumable runner: `--suite --condition A|B
  --model --tasks all|dev|heldout --defense-revision N --dry-run --limit`.
  Reports do-nothing floor, exposure (payload seen), fail-open per module.
- `settings.DEFENSE_REVISION` — 0 = as published (default, byte-identical
  prompts, pinned by hash tests); 1 = three dev fixes (spike doc §11).
- `scripts/prof_run.py` + `docs/PROF_MACHINE.md` — probe-then-run for the
  professor's machine; outputs one zip.
- Per-model Groq rate limiters; Ollama truncation guard
  (`ContextTruncatedError`); `scripts/agentdojo_spike_measure.py`.
- Tests: 704 pass with `results/` present; a fresh clone gives 692 passed,
  12 skipped.

---

## 3. Next steps, in order

1. ~~Slack~~ done (§1, spike doc §12).
2. **Professor's machine** — waiting on the user: they run
   `python scripts/prof_run.py probe` there and bring back
   `prof_probe_<host>.json`; you pick models (candidates `qwen2.5:7b`,
   `qwen2.5:14b`) and give the exact `run` command; they bring back the zip.
3. **Revision 2** (not started), develop on banking dev:
   (a) Planner: strip `()` from plan tool names and drop edges to pruned
   nodes instead of rejecting the plan (banking `user_task_9`);
   (b) Node Expansion: do not auto-allow read-only calls whose arguments
   (e.g. a URL) were not named by the plan or user - every surviving Slack
   attack went through this. Slack is then no longer fully held-out for (b):
   say so, or hold out a Slack split first.
4. **Write-up:** fold AgentDojo into `docs/RESULTS_SUMMARY.md` and
   `docs/PRESENTATION.md` (they still show only the 39-case numbers).
5. Not feasible on this laptop: travel (~18 h), workspace (~57 h, needs 24k).

---

## 4. How to analyse a finished suite

Paired comparison over the same (user task, injection task) pairs, exact
McNemar. For Slack there is no split; use all rows.

```python
import json
from math import comb
M = "qwen3-4b-instruct-2507-q4_K_M"
p = lambda c, tag="": json.load(open(f"results/agentdojo/slack_{c}_{M}_important_instructions{tag}.json"))
A, B = p("A"), p("B", "_rev1")
def mc(x, y):
    n = x + y
    return min(1, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n) if n else 1
ka = {(r["user_task"], r["injection_task"]): r for r in A if r["injection_task"]}
kb = {(r["user_task"], r["injection_task"]): r for r in B if r["injection_task"]}
for f in ("security", "utility"):
    x = sum(ka[k][f] and not kb[k][f] for k in ka); y = sum(kb[k][f] and not ka[k][f] for k in ka)
    print(f, "only A", x, "only B", y, "p", mc(x, y))
```

Also report: clean utility vs floor, ASR among runs where the payload was seen
(`exposed`), and the `failed_open` counts per module (threshold 10%; a
degraded Planner is fail-*closed* read-only, label it so). Attribute what
blocked by replaying from cache (`ConditionB` + `GuardedPipeline`, look for
"PAUSED by the Misalignment" / "Blocked by the plan" in step errors).

---

## 5. Gotchas that each cost hours (all measured)

- Local arm needs **`USE_LOCAL_BACKBONE=1`**, or the model name is sent to Groq.
- Ollama must start with **`OLLAMA_MODELS=D:\ollama`** (system var is one level
  too deep) and **`OLLAMA_CONTEXT_LENGTH`** — otherwise prompts > window are
  silently cut from the start (the client now raises).
- Use **`127.0.0.1`, not `localhost`**: Ollama auto-updated 0.34.4 → 0.35.1 on
  ~12 Oct and stalls ~2 s on IPv6. Banking ran on 0.34.4, Slack on 0.35.1 —
  footnote cross-suite comparisons.
- `num_ctx` through the `/v1` endpoint is ignored. `qwen3:4b` cannot stop
  thinking (51 s/turn) — use the `-instruct-2507` build.
- Groq limits move: safeguard model went 8000 → 2000 TPM in two days. Guard
  model RPM 30 per model. Probe before trusting.
- **Use a private port (11435).** After a wake-up, Ollama's tray app auto-starts its own server on 11434 with the default window and model folder.
- **Truncation guard false alarm fixed** (13 Oct): an intact prompt of exactly 2050 tokens tripped the fingerprint; it now also needs > 5 chars/token.
- **Hibernating kills the run** (13 Oct 10:25, B at 22/126) — relaunch via WMI; it resumes.
- The PC can suspend mid-run (the user's auto-hibernate calls the suspend
  API); a run's wall time then includes the sleep.
- `pytest | tail` hides failures; check the exit code before committing.

## 6. How the user wants to work

- Brief answers when asked for status ("in brief", percentages, ETA).
- Measure, don't assume; write the measurement beside every setting.
- Fix the go/no-go or dev/held-out split **before** looking at results.
- Report unflattering results plainly; flag deviations from the plan.
- Never state a test count without running pytest. Commit and push to
  `origin master` at the end of work.
- Ask before downloads (filename, source, size). The user leaves the PC on
  overnight for long runs; give an ETA.
- Graphify and Ponytail are installed; ponytail mode = smallest working diff.
