# Paper draft

`main.tex` (ACL long-paper format), `references.bib`, generated `tables/` and `figures/`.

## Build

1. Regenerate tables and figures from the raw results (gitignored, read-only):

   ```bash
   python paper/scripts/make_figures.py D:/Project/MinorProject/.claude/worktrees/project-readme-comprehensive-47c733/results/agentdojo
   ```

   It prints every count it uses. It needs `matplotlib` only; the Fisher test is stdlib.
2. Compile with the official ACL style (`acl.sty`, `acl_natbib.bst` from
   https://github.com/acl-org/acl-style-files, or Overleaf's ACL template): pdflatex, bibtex, pdflatex ×2.

## Claims and their sources

"Script" means `paper/scripts/make_figures.py` run on the result files in
`results/agentdojo/` (file names listed in its `FILES` table). "Spike" means
`docs/AGENTDOJO_SPIKE.md` on this branch. "Attribution" means §21 of
`docs/AGENTDOJO_SPIKE.md` on the unpushed branch `claude/goofy-morse-1a8257`; its
section number clashes with this branch's §21.

| # | Claim in the paper | Source |
|---|---|---|
| 1 | 4B ASR 39/72→0/72 (banking held-out), 37/55→8/55 (Slack held-out), 15/140→0/140 (travel), p<0.0001 each | Script → `tables/main_rows.tex`; Spike §11, §16, §19 |
| 2 | 7B ASR 27→0, 23→4, travel 10→3 (p=0.085) | Script; Spike §19, §21 |
| 3 | Rev 5 UA not significantly below no defense on any row (p≥0.11) | Script (`main_rows.tex` UA p column) |
| 4 | Rev 1 (dev-only) banking held-out 0/72 ASR, UA 32/72 | Script (`rev 1` row printout); Spike §11.1 |
| 5 | Rev 3 7B Slack held-out UA 22→9 (p=0.010) | Spike §21 test 3 |
| 6 | Rev 5 vs rev 3 banking held-out UA excl. tasks 8,10: 26/54 vs 14/54 (p=0.028) on 4B; 20/54 vs 12/54 (p=0.14) on 7B; 7B banking dev 18 vs 28 | Spike §16, §21 test 1 and "not pre-registered" |
| 7 | Travel rev 5 vs rev 3 UA 26 vs 23 (p=0.75, 4B), 10 vs 6 (p=0.44, 7B) | Spike §19 Q1 |
| 8 | 7B saw travel payload in 56/140 undefended runs | Script (`seen`); Spike §19 |
| 9 | Tool filter 6/72, PI detector 6/72, spotlighting 39/72, repeat-prompt 24/60 (68/80 runs); guard 0/72 vs filter/detector p=0.028 | Script → `tables/baselines_rows.tex`. The p-value is computed by the script and is **not pre-registered** (Spike §15.3 planned paired McNemar vs A, rev 1, rev 3) |
| 10 | Ablation: Planner-only 5/72, 9/55; Checkpoint-only 0/72, 7/55; Firewall-only 34/72, 28/55; pre-registered rule not met | Script → `tables/ablation_rows.tex`; Spike §22–23 |
| 11 | Full ensemble the only config above A on banking UA (44 vs 31, p=0.045); Checkpoint-only Slack UA 9 vs 22 (p=0.010) | Spike §23 |
| 12 | Hardware drift: Slack A 37 vs 28 (p=0.12) | Script; Spike §23 |
| 13 | Planner-only banking attacks all on user tasks 6 and 10; Checkpoint-only and full-ensemble Slack residuals all `injection_task_3` | Spike §23 "not pre-registered"; residual lists printed by Script |
| 14 | 7B travel residual: 3 attacks, all `injection_task_6` (a "say X" goal with no tool call) | Script residual printout; goal text from `agentdojo/default_suites/v1/travel/injection_tasks.py` |
| 15 | Block attribution table (rev 5, 4B): banking 36/4/0/23/9, Slack 27/1/4/10/5, travel 3/0/24/29/21 of 77 | Attribution §21 |
| 16 | Quarantine ran in 50/72 stopped banking runs without removing the payload; travel's 24 Firewall stops are the e-mail exfiltration goals | Attribution §21 |
| 17 | Rev 3 held-out loss attribution: banking 11 (9 Checkpoint, 2 Planner); Slack 13 (5 empty plans, 2 Checkpoint, 6 agent) | Spike §16 |
| 18 | Rev 2 Slack held-out: ASR 0/55, UA 2/55; 9 of 10 lost runs from the address rule | Script; Spike §13.2 |
| 19 | Model calls per run, 1.6–2.1× | Script → `tables/cost_rows.tex` |
| 20 | Wall-clock unusable: defended runs all have cache hits | Script printout (0 uncached defended runs) |
| 21 | In-house latency +738 ms (27B), +3025 ms (3B) per tool call | `docs/RESULTS_SUMMARY.md` §4b |
| 22 | In-house 3B: ASR_inj 6/15→0/15 (p=0.008), HS 8/8→0/8 (p=0.0001), one-sided; 27B 1/15, 2/8 baselines n.s.; MF1 0.81 | `docs/RESULTS_SUMMARY.md` §2, §4a |
| 23 | In-house GAI 0.933 vs 0.817 best single module; 27B GAI with LAT 0.927→0.830 | `docs/RESULTS_SUMMARY.md` §1, §4a |
| 24 | Rev 5 in-house MF1 0.76→0.90 on 21 cases; false-premise case now skipped | Spike §16 |
| 25 | Temperature 0.7 repeats: A ASR 0.33–0.53, guard 0.00; BU 0.80, over-refusal 0.077 | `docs/RESULTS_SUMMARY.md` §2a |
| 26 | Rev 0 as built: ASR 60/144→1/144; clean 9/16→6/16, 5 of 6 floor tasks | Spike §10 |
| 27 | qwen2.5:3b failed go/no-go (7/16 vs bar 8/16, floor 5/16) | Spike §9.1 |
| 28 | Rev 5 banking gain on tasks 14 (0→9 of 9) and 13 (5→8 of 9) | Spike §16 |
| 29 | Travel task 19 planner truncation in all 8 defended runs, counted | Spike §18 amendment 2; Script failed-open count |
| 30 | Workspace not run: hosted TPM limit; firewall flagged 28/84 clean outputs | Spike §4.3, §4.8 |
| 31 | Source papers' numbers (Table 1) | `CLAUDE.md` §9.2 and the PDFs in `docs/papers` |
| 32 | IPIGuard App. G includes a Qwen2.5-7B-Instruct executor | IPIGuard PDF, Table 5 |

## Open items

- `TODO(author)` in `main.tex`: supervisor or course line, if wanted.
- Page count against the 8-page limit is not yet checked against a compiled PDF.
