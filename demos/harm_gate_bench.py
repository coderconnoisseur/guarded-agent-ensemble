"""Harm Gate benchmarked against AgentHarm's 352 paired behaviours.

    python scripts/fetch_agentharm.py      # once, downloads the data
    python demos/harm_gate_bench.py        # rubric only - ZERO LLM calls
    python demos/harm_gate_bench.py --classifier --limit 40   # + stage 2

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
This is a **gate-level** benchmark, not `HS`. AgentHarm's tasks call its own
~100 synthetic tools, so they are not executable in our sandbox and we are not
running AgentHarm. What we are doing is feeding its prompts to the Harm Gate,
which is a task-intake classifier that decides refuse-or-allow from the
request text before any tool is involved.

`HS` in the GAI stays what it was: agent-level compliance, graded from real
tool calls over our own direct-harm suite. Conflating the two would be
exactly the kind of quiet metric drift this project keeps finding in itself.

WHY IT IS WORTH RUNNING ANYWAY
------------------------------
Our own direct-harm suite has 8 harmful cases. Measured 2026-09-20, that is
not enough to distinguish the Harm Gate from doing nothing: `HS` 0.25 -> 0.00
carries a Fisher p of 0.233. AgentHarm gives 176 harmful behaviours and 176
benign counterparts across 8 categories - a denominator 22x larger.

And it is nearly free. The Harm Gate's stage 1 is a deterministic rubric with
**no LLM call at all**, so all 352 prompts can be scored for zero requests.
Stage 2 escalates to a dedicated safety model only when the rubric is
uncertain, so `--classifier` costs one request per escalation, not per case.

The pairing is what makes it honest: each harmful behaviour has a benign
counterpart in the same category, so detection and over-refusal are read off
the same axis. A gate that flags everything scores perfect detection and is
caught immediately by the benign column.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.fetch_agentharm import is_fetched, load_cached  # noqa: E402
from src.defense.harm_gate import HarmGate  # noqa: E402
from src.eval.stats import fisher_exact_one_sided, format_rate  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402

RULE = "=" * 92


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def evaluate(gate: HarmGate, rows: list[dict]) -> list[tuple[dict, bool, str]]:
    """(behaviour, flagged, stage) for each prompt."""
    out = []
    for row in rows:
        verdict = gate.check(row["prompt"])
        out.append((row, verdict.flagged, verdict.stage))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Harm Gate vs AgentHarm.")
    parser.add_argument("--classifier", action="store_true",
                        help="Enable stage 2. Costs one request per escalation.")
    parser.add_argument("--limit", type=int,
                        help="Only the first N behaviours per class.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)

    if not is_fetched():
        print("AgentHarm is not downloaded yet.")
        print("Run:  python scripts/fetch_agentharm.py")
        return 1

    harmful = load_cached("harmful")
    benign = load_cached("benign")
    if args.limit:
        harmful, benign = harmful[:args.limit], benign[:args.limit]

    with LLMClient() as client:
        gate = HarmGate(client=client, enable_classifier=args.classifier)

        banner("HARM GATE vs AgentHarm (gate-level, NOT the GAI's HS)")
        print(f"  Behaviours : {len(harmful)} harmful, {len(benign)} benign "
              f"counterparts")
        print(f"  Stage 2    : {'ON (costs requests)' if args.classifier else 'OFF - zero LLM calls'}")
        print("  These prompts call AgentHarm's own tools and are NOT executable")
        print("  in our sandbox. This measures the intake classifier only.")

        harmful_results = evaluate(gate, harmful)
        benign_results = evaluate(gate, benign)

        caught = sum(1 for _, flagged, _ in harmful_results if flagged)
        over = sum(1 for _, flagged, _ in benign_results if flagged)

        banner("RESULTS")
        print(f"  harmful flagged (detection)  {format_rate(caught, len(harmful))}")
        print(f"  benign flagged (over-refusal){format_rate(over, len(benign))}")
        p = fisher_exact_one_sided(over, len(benign), caught, len(harmful))
        print(f"\n  Fisher exact, benign vs harmful flag rate: p={p:.2e} "
              f"{'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")
        print("  (does the gate separate the two classes at all?)")

        banner("BY CATEGORY")
        print(f"  {'category':16} {'detected':>26} {'over-refused':>26}")
        cats = sorted({r.get("category", "?") for r in harmful})
        for cat in cats:
            h = [(r, f) for r, f, _ in harmful_results if r.get("category") == cat]
            b = [(r, f) for r, f, _ in benign_results if r.get("category") == cat]
            print(f"  {cat:16} "
                  f"{format_rate(sum(1 for _, f in h if f), len(h)):>26} "
                  f"{format_rate(sum(1 for _, f in b if f), len(b)):>26}")

        banner("WHICH STAGE DECIDED")
        stages = Counter(stage for _, _, stage in harmful_results + benign_results)
        for stage, count in stages.most_common():
            print(f"  {stage:12} {count}")

        banner("READ THIS BEFORE QUOTING THE NUMBER")
        print("  Detection here is the gate flagging a REQUEST. It is not HS,")
        print("  which is agent-level compliance graded from real tool calls on")
        print("  our own suite. A high number here and a low HS are compatible:")
        print("  the gate can recognise harm it has no tools to commit.")
        print()
        print("  The benign column is the one that constrains. AgentHarm pairs")
        print("  every harmful behaviour with a benign counterpart precisely so")
        print("  a refuse-everything gate cannot look good.")
        print(f"\n  Budget: {client.budget_summary()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
