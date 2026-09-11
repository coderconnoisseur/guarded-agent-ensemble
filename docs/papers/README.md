# Source papers

The six papers this project adapts are **not committed**. They are other
people's copyrighted work, this repository is public, and their redistribution
licences have not been individually verified — so `docs/papers/*.pdf` is
gitignored. Every one of them is freely available from its publisher.

| # | Paper | Venue | Module it informs |
|---|---|---|---|
| 1 | InferAct: Inferring Safe Actions via Theory-of-Mind | EMNLP 2025 | Misalignment Checkpoint (Phase 5) |
| 2 | IPIGuard: Plan-Then-Execute against Indirect Prompt Injection | EMNLP 2025 | Planner / Tool Dependency Graph (Phase 3) |
| 3 | AgentVigil: Automated Black-Box Red-Teaming | EMNLP 2025 Findings | Diversity suite (stretch, §11) |
| 4 | AgentHarm: Measuring Harmfulness of LLM Agents | ICLR 2025 | Harm Gate + direct-harm suite (Phase 2) |
| 5 | SIRAJ: Diversity-Optimized Red-Teaming & Distillation | EACL 2026 | Distillation (stretch, §11) |
| 6 | ShieldMCP: Runtime Defense for the Model Context Protocol | ACL 2026 Industry | Response Firewall + Quarantine (Phase 4) |

Where to find them: ACL Anthology (`aclanthology.org`) for the EMNLP, EACL and
ACL entries, OpenReview for the ICLR entry, or arXiv for preprints.

**You do not need the PDFs to build or run this project.** `CLAUDE.md` §3
contains a condensed technical reference for each paper's mechanism — what it
does and which module adapts it — and that is what the code was written
against. The PDFs are for checking our reading of a mechanism against the
source, and for the final report's citations.

Drop them in this folder if you want them locally; git will ignore them.
