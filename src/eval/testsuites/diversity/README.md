# diversity/ — intentionally empty

This suite holds no test cases, and that is a scoping decision rather than an
oversight.

`DIV_ASR` (CLAUDE.md §9) is defined as the attack success rate on a
**self-generated, adaptive** corpus: an AgentVigil/SIRAJ-style loop of seed
templates, an LLM mutator, and success/coverage-driven seed selection (§3,
§11). Its whole purpose is to test whether the ensemble *generalises* to
attacks it has never seen, as opposed to the three fixed suites beside it.

Hand-writing cases here would defeat that. They would be authored by the same
person who wrote the defenses, drawn from the same handful of attack shapes as
`injection/`, and the resulting number would look like a generalisation
measurement while actually measuring nothing of the kind — the single most
misleading thing this project could put in front of a reviewer.

## How this is handled instead

§9 already prescribes the honest options when the diversity suite is not
built: state the omission explicitly in the report, and either drop the
`DIV_ASR` term and renormalise the remaining six weights, or carry it as a
clearly flagged placeholder. Phase 6's `eval/report.py` does that; it does not
silently omit the term.

§11 lists building this suite for real as a stretch goal, and §13 already
names it "the most likely thing to be cut for time".

## To populate it later

Build the mutation loop described in §3 (AgentVigil) — seed corpus, LLM-based
mutator, greedy or round-robin seed selection, MCTS optional — and have it
emit cases in the same §8.1 schema as the other three suites. The runner needs
no changes: `load_suites` already treats an empty suite directory as valid, so
files appearing here start counting automatically.
