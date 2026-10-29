"""Regenerate every figure and data table in the paper from the raw AgentDojo result files.

Usage:
    python paper/scripts/make_figures.py [RESULTS_DIR]

RESULTS_DIR defaults to $RESULTS_DIR, else ../results/agentdojo relative to the repo root.
Result files are gitignored; on the author's machine they live in
D:/Project/MinorProject/.claude/worktrees/project-readme-comprehensive-47c733/results/agentdojo.

Writes paper/figures/*.pdf and paper/tables/*.tex (generated; do not edit by hand),
and prints every count it used so each number in the paper can be checked.
Nothing is estimated: counts are read from the rows, p-values are two-sided Fisher
exact tests computed from those counts.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
FIG, TAB = PAPER / "figures", PAPER / "tables"
REPO_RESULTS = PAPER.parent / "results" / "agentdojo"

# Committed splits (src/agentdojo_bridge.py). Travel has no split: all of it was unseen.
BANKING_HELDOUT = {f"user_task_{i}" for i in (3, 4, 6, 8, 10, 12, 13, 14)}
SLACK_HELDOUT = {f"user_task_{i}" for i in (1, 3, 4, 5, 6, 8, 9, 12, 14, 17, 19)}
HELDOUT = {"banking": BANKING_HELDOUT, "slack": SLACK_HELDOUT}

Q4 = "qwen3-4b-instruct-2507-q4_K_M_important_instructions"
Q7 = "qwen2.5-7b_important_instructions"
PL = "_plan-qwen-qwen3.8-27b"

# (key) -> (file, split filter). split filter None = take all rows of the file.
# The 4B laptop A runs were made on all tasks and are filtered to held-out here;
# the *_A_..._heldout.json files are the Kaggle re-runs (hardware drift check, spike 23).
FILES = {
    # 4B, banking held-out
    ("4B", "banking", "A"): (f"banking_A_{Q4}.json", "heldout"),
    ("4B", "banking", "A (Kaggle)"): (f"banking_A_{Q4}_heldout.json", None),
    ("4B", "banking", "rev 0"): (f"banking_B_{Q4}.json", "heldout"),
    ("4B", "banking", "rev 1"): (f"banking_B_{Q4}_heldout_rev1.json", None),
    ("4B", "banking", "rev 2"): (f"banking_B_{Q4}_heldout_rev2.json", None),
    ("4B", "banking", "rev 3"): (f"banking_B_{Q4}_heldout_rev3{PL}.json", None),
    ("4B", "banking", "rev 5"): (f"banking_B_{Q4}_heldout_rev5{PL}.json", None),
    ("4B", "banking", "Planner only"): (f"banking_B_{Q4}_heldout_rev5{PL}_mod-planner.json", None),
    ("4B", "banking", "Firewall only"): (f"banking_B_{Q4}_heldout_rev5{PL}_mod-firewall+quarantine.json", None),
    ("4B", "banking", "Checkpoint only"): (f"banking_B_{Q4}_heldout_rev5{PL}_mod-misalignment.json", None),
    ("4B", "banking", "Tool filter"): (f"banking_tool_filter_{Q4}_heldout.json", None),
    ("4B", "banking", "PI detector"): (f"banking_transformers_pi_detector_{Q4}_heldout.json", None),
    ("4B", "banking", "Spotlighting"): (f"banking_spotlighting_with_delimiting_{Q4}_heldout.json", None),
    ("4B", "banking", "Repeat prompt"): (f"banking_repeat_user_prompt_{Q4}_heldout.json", None),
    # 4B, Slack held-out
    ("4B", "slack", "A"): (f"slack_A_{Q4}.json", "heldout"),
    ("4B", "slack", "A (Kaggle)"): (f"slack_A_{Q4}_heldout.json", None),
    ("4B", "slack", "rev 1"): (f"slack_B_{Q4}_rev1.json", "heldout"),
    ("4B", "slack", "rev 2"): (f"slack_B_{Q4}_heldout_rev2.json", None),
    ("4B", "slack", "rev 3"): (f"slack_B_{Q4}_heldout_rev3{PL}.json", None),
    ("4B", "slack", "rev 5"): (f"slack_B_{Q4}_heldout_rev5{PL}.json", None),
    ("4B", "slack", "Planner only"): (f"slack_B_{Q4}_heldout_rev5{PL}_mod-planner.json", None),
    ("4B", "slack", "Firewall only"): (f"slack_B_{Q4}_heldout_rev5{PL}_mod-firewall+quarantine.json", None),
    ("4B", "slack", "Checkpoint only"): (f"slack_B_{Q4}_heldout_rev5{PL}_mod-misalignment.json", None),
    # travel (all unseen)
    ("4B", "travel", "A"): (f"travel_A_{Q4}.json", None),
    ("4B", "travel", "rev 3"): (f"travel_B_{Q4}_rev3{PL}.json", None),
    ("4B", "travel", "rev 5"): (f"travel_B_{Q4}_rev5{PL}.json", None),
    ("7B", "travel", "A"): (f"travel_A_{Q7}.json", None),
    ("7B", "travel", "rev 3"): (f"travel_B_{Q7}_rev3{PL}.json", None),
    ("7B", "travel", "rev 5"): (f"travel_B_{Q7}_rev5{PL}.json", None),
    # 7B, banking / Slack held-out
    ("7B", "banking", "A"): (f"banking_A_{Q7}_heldout.json", None),
    ("7B", "banking", "rev 3"): (f"banking_B_{Q7}_heldout_rev3{PL}.json", None),
    ("7B", "banking", "rev 5"): (f"banking_B_{Q7}_heldout_rev5{PL}.json", None),
    ("7B", "slack", "A"): (f"slack_A_{Q7}_heldout.json", None),
    ("7B", "slack", "rev 3"): (f"slack_B_{Q7}_heldout_rev3{PL}.json", None),
    ("7B", "slack", "rev 5"): (f"slack_B_{Q7}_heldout_rev5{PL}.json", None),
}


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p for successes a/n1 vs b/n2 (sum of tables no likelier than observed)."""
    k, n = a + b, n1 + n2
    denom = comb(n, k)
    p_obs = comb(n1, a) * comb(n2, b) / denom
    lo, hi = max(0, k - n2), min(k, n1)
    p = sum(comb(n1, x) * comb(n2, k - x) / denom for x in range(lo, hi + 1)
            if comb(n1, x) * comb(n2, k - x) / denom <= p_obs * (1 + 1e-9))
    return min(1.0, p)


def load(results: Path, name: str, split: str | None, suite: str) -> list[dict]:
    rows = json.loads((results / name).read_text(encoding="utf-8"))
    if split == "heldout":
        rows = [r for r in rows if r["user_task"] in HELDOUT[suite]]
    return rows


def summarise(rows: list[dict]) -> dict:
    att = [r for r in rows if r.get("injection_task")]
    clean = [r for r in rows if not r.get("injection_task")]
    uncached = [r["wall_ms"] for r in rows if r.get("num_cache_hits", 0) == 0 and "wall_ms" in r]
    return {
        "runs": len(rows), "n_att": len(att), "n_clean": len(clean),
        "asr": sum(1 for r in att if r.get("security")),
        "ua": sum(1 for r in att if r.get("utility")),
        "clean": sum(1 for r in clean if r.get("utility")),
        "seen": sum(1 for r in att if r.get("exposed")),
        "calls": statistics.mean(r["num_llm_calls"] for r in rows),
        "wall_s_median_uncached": statistics.median(uncached) / 1000 if uncached else None,
        "n_uncached": len(uncached),
        "failed_open": sum(1 for r in rows if any((r.get("failed_open") or {}).values())),
        "inj": [r["injection_task"] for r in att if r.get("security")],
    }


def pfmt(p: float) -> str:
    if p < 1e-4:
        return "$<$0.0001"
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def frac(x: int, n: int) -> str:
    return f"{x}/{n}"


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", REPO_RESULTS))
    S = {k: summarise(load(results, f, split, k[1])) for k, (f, split) in FILES.items()}

    print("key | runs | ASR | UA | clean | seen | calls/run | median wall s (uncached, n) | failed-open")
    for k, s in S.items():
        print(f"{k} | {s['runs']} | {s['asr']}/{s['n_att']} | {s['ua']}/{s['n_att']} | "
              f"{s['clean']}/{s['n_clean']} | {s['seen']} | {s['calls']:.2f} | "
              f"{s['wall_s_median_uncached']} ({s['n_uncached']}) | {s['failed_open']}")
    for k in [("4B", "slack", "rev 5"), ("7B", "slack", "rev 5"), ("4B", "slack", "Checkpoint only"),
              ("4B", "slack", "Planner only"), ("7B", "travel", "rev 5")]:
        print("residual", k, sorted(set(S[k]["inj"])), len(S[k]["inj"]))

    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    # Palette: neutral grey for the undefended baseline, two blues for the two revisions.
    C = {"A": "#8c8c8c", "rev 3": "#9ecae1", "rev 5": "#2171b5"}

    # ---- Table: main results -------------------------------------------------
    lines = []
    for bb in ("4B", "7B"):
        for suite, label in (("banking", "Banking (held-out)"), ("slack", "Slack (held-out)"),
                             ("travel", "Travel (unseen)")):
            a = S[(bb, suite, "A")]
            cells = []
            for cond in ("A", "rev 3", "rev 5"):
                s = S[(bb, suite, cond)]
                cells += [frac(s["asr"], s["n_att"]), frac(s["ua"], s["n_att"]), frac(s["clean"], s["n_clean"])]
            r5 = S[(bb, suite, "rev 5")]
            p_asr = fisher_two_sided(a["asr"], a["n_att"], r5["asr"], r5["n_att"])
            p_ua = fisher_two_sided(a["ua"], a["n_att"], r5["ua"], r5["n_att"])
            lines.append(f"{bb} & {label} & " + " & ".join(cells) + f" & {pfmt(p_asr)} & {pfmt(p_ua)} \\\\")
        if bb == "4B":
            lines.append("\\midrule")
    (TAB / "main_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table: AgentDojo's built-in defenses, banking held-out, 4B ----------
    lines = []
    full = S[("4B", "banking", "rev 5")]
    for cond in ("A", "Spotlighting", "Repeat prompt", "Tool filter", "PI detector", "rev 3", "rev 5"):
        s = S[("4B", "banking", cond)]
        p = fisher_two_sided(s["asr"], s["n_att"], full["asr"], full["n_att"]) if cond != "rev 5" else None
        name = {"rev 3": "Ensemble rev 3", "rev 5": "Ensemble rev 5"}.get(cond, cond)
        if cond == "A":
            name = "No defense"
        note = f" ({s['runs']}/80 runs)" if s["runs"] != 80 else ""
        lines.append(f"{name}{note} & {frac(s['asr'], s['n_att'])} & {frac(s['ua'], s['n_att'])} & "
                     f"{frac(s['clean'], s['n_clean'])} & {pfmt(p) if p is not None else '--'} \\\\")
    (TAB / "baselines_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table: single-module ablation (4B held-out) -------------------------
    lines = []
    for suite, label in (("banking", "Banking"), ("slack", "Slack")):
        full = S[("4B", suite, "rev 5")]
        ak = S[("4B", suite, "A (Kaggle)")]
        for cond in ("A", "A (Kaggle)", "Planner only", "Checkpoint only", "Firewall only", "rev 5"):
            s = S[("4B", suite, cond)]
            single = cond.endswith("only")
            p_full = pfmt(fisher_two_sided(s["asr"], s["n_att"], full["asr"], full["n_att"])) if single else "--"
            p_a = pfmt(fisher_two_sided(s["asr"], s["n_att"], ak["asr"], ak["n_att"])) if single else "--"
            name = {"A": "No defense (laptop)", "A (Kaggle)": "No defense (Kaggle)",
                    "rev 5": "\\textbf{Full ensemble}", "Firewall only": "Firewall + Quarantine only"}.get(cond, cond)
            lines.append(f"{label if cond == 'A' else ''} & {name} & {frac(s['asr'], s['n_att'])} & "
                         f"{frac(s['ua'], s['n_att'])} & {frac(s['clean'], s['n_clean'])} & {p_full} & {p_a} \\\\")
        if suite == "banking":
            lines.append("\\midrule")
    (TAB / "ablation_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table: cost ---------------------------------------------------------
    # Model calls per run only. wall_ms is not usable as a latency measure here: every
    # defended run had LLM-cache hits, A and B ran on different machines for some suites,
    # and hosted calls include rate-limiter waits. Latency comes from the in-house timing
    # runs instead (docs/RESULTS_SUMMARY.md 4b).
    lines = []
    for bb in ("4B", "7B"):
        for suite, label in (("banking", "Banking"), ("slack", "Slack"), ("travel", "Travel")):
            a, b = S[(bb, suite, "A")], S[(bb, suite, "rev 5")]
            lines.append(f"{bb} & {label} & {a['calls']:.1f} & {b['calls']:.1f} & {b['calls'] / a['calls']:.1f}$\\times$ \\\\")
    (TAB / "cost_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Figure: ASR by configuration ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.9), sharey=True)
    suites = [("banking", "Banking"), ("slack", "Slack"), ("travel", "Travel")]
    for ax, bb in zip(axes, ("4B", "7B")):
        for i, cond in enumerate(("A", "rev 3", "rev 5")):
            xs = [j + (i - 1) * 0.27 for j in range(3)]
            ys = [100 * S[(bb, s, cond)]["asr"] / S[(bb, s, cond)]["n_att"] for s, _ in suites]
            ax.bar(xs, ys, width=0.25, color=C[cond], label={"A": "No defense"}.get(cond, cond),
                   edgecolor="white", linewidth=0.3)
            for x, y, (s, _) in zip(xs, ys, suites):
                ax.text(x, y + 1, str(S[(bb, s, cond)]["asr"]), ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(range(3), [l for _, l in suites])
        ax.set_title(f"Qwen {'3 4B' if bb == '4B' else '2.5 7B'}", fontsize=8)
        ax.tick_params(length=2)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Attack success (%)")
    axes[1].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "asr_by_config.pdf")
    plt.close(fig)

    # ---- Figure: security-utility trade-off (4B held-out) -------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
    style = {  # marker, colour
        "A": ("o", "#8c8c8c"), "A (Kaggle)": ("o", "#bdbdbd"),
        "rev 0": ("s", "#c6dbef"), "rev 1": ("s", "#9ecae1"), "rev 2": ("s", "#6baed6"),
        "rev 3": ("s", "#4292c6"), "rev 5": ("s", "#08519c"),
        "Planner only": ("^", "#e6550d"), "Checkpoint only": ("^", "#fd8d3c"),
        "Firewall only": ("^", "#fdbe85"),
        "Tool filter": ("D", "#31a354"), "PI detector": ("D", "#74c476"),
        "Spotlighting": ("D", "#a1d99b"),
    }
    # Label placement only (dx, dy in points, alignment); positions come from the data.
    offsets = {
        ("banking", "A"): (5, -2, "left"), ("banking", "A (Kaggle)"): (5, 3, "left"),
        ("banking", "Spotlighting"): (-5, -3, "right"), ("banking", "Planner only"): (-4, 3, "right"),
        ("banking", "Tool filter"): (4, 3, "left"),
        ("slack", "A (Kaggle)"): (-5, 0, "right"), ("slack", "Firewall only"): (4, 2, "left"),
        ("slack", "Checkpoint only"): (4, -8, "left"),
    }
    for ax, suite, title in ((axes[0], "banking", "Banking held-out (72 attacked runs)"),
                             (axes[1], "slack", "Slack held-out (55 attacked runs)")):
        labels: dict[tuple[float, float], list[str]] = {}
        for cond, (m, c) in style.items():
            k = ("4B", suite, cond)
            if k not in S:
                continue
            s = S[k]
            x, y = 100 * s["ua"] / s["n_att"], 100 * s["asr"] / s["n_att"]
            ax.scatter(x, y, marker=m, color=c, s=22, edgecolor="black", linewidth=0.3, zorder=3)
            labels.setdefault((x, y), []).append(cond)
        for (x, y), names in labels.items():  # identical points share one label
            dx, dy, ha = offsets.get((suite, names[0]), (3, 2, "left")) if len(names) == 1 else (4, -9, "left")
            text = ", ".join(names)
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.5, ha=ha)
        ax.set_ylim(bottom=-7)  # room for labels under the y = 0 points
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ax.get_ylim()[1]])
        ax.set_xlabel("Utility under attack (%)  →  better")
        ax.set_title(title, fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Attack success (%)  ↓  better")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)
    print("wrote", sorted(p.name for p in FIG.iterdir()), sorted(p.name for p in TAB.iterdir()))


if __name__ == "__main__":
    # Self-check of the Fisher test against known values (scipy.stats.fisher_exact, two-sided).
    assert abs(fisher_two_sided(1, 4, 3, 4) - 0.4857142857) < 1e-6
    assert abs(fisher_two_sided(8, 8, 0, 8) - 0.0001554002) < 1e-6
    main()
