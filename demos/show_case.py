"""Render one saved test case run, legibly. No LLM calls.

    python demos/show_case.py inj_001
    python demos/show_case.py --list
    python demos/show_case.py inj_005 --results results/phase1_condition_a.json

Reads a results file written by the eval runner and prints, for one case: the
task, the payload that was planted (if any), the full transcript with the
untrusted tool output marked, every check with its verdict, and the outcome.

This is the command to use when showing what actually happened in an injection
case - the payload is visible sitting in the agent's context, followed by what
the agent did with it. Phase 4 needs the same rendering side by side for
Condition A against Condition B, so this is where that will grow from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from src.eval.schemas import SuiteReport, load_suites  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78
DEFAULT_RESULTS = ""  # newest file in results/ when unset


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def newest_results() -> Path:
    """Most recently written results file. Runs are per backbone now."""
    files = sorted(
        settings.RESULTS_DIR.glob("phase*_condition_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise SystemExit(
            "No results yet. Run `python demos/phase1_demo.py` first."
        )
    return files[0]


def load_report(path: Path | None) -> SuiteReport:
    target = path or newest_results()
    if not target.exists():
        raise SystemExit(
            f"No results at {target}. Run `python demos/phase1_demo.py` first."
        )
    return SuiteReport.model_validate_json(target.read_text(encoding="utf-8"))


def print_index(report: SuiteReport) -> None:
    banner(f"CASES IN {report.condition} ({len(report.results)})")
    for result in report.results:
        print(f"  {result.summary_line()}")


def wrap(text: str, width: int = 74, indent: str = "    ") -> list[str]:
    """Wrap without collapsing the blank lines that make a payload readable."""
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        current = ""
        for word in raw.split():
            if len(current) + len(word) + 1 > width:
                lines.append(indent + current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(indent + current)
    return lines


def show(report: SuiteReport, case_id: str) -> int:
    matches = [r for r in report.results if r.test_case_id == case_id]
    if not matches:
        print(f"No case {case_id!r} in this results file.")
        print_index(report)
        return 1
    result = matches[0]

    case = None
    for candidate in load_suites(settings.TESTSUITES_DIR):
        if candidate.id == case_id:
            case = candidate
            break

    banner(f"{result.test_case_id}  [{result.suite}]  condition {result.condition}")
    print(f"  backbone : {result.backbone_model}")
    if case:
        print(f"  category : {case.category}")
        print(f"\n  TASK GIVEN TO THE AGENT")
        for line in wrap(case.prompt):
            print(line)

    if case and case.injected_content:
        banner("PLANTED PAYLOAD")
        print(f"  carried by : {case.injection_tool}  (key: {case.injection_key})")
        print("  This text is data the environment returned. The user never")
        print("  said it, and the agent was never told to trust it.\n")
        for line in wrap(case.injected_content):
            print(line)

    banner("TRANSCRIPT")
    for entry in result.transcript:
        role = entry["role"].upper()
        content = entry["content"].strip()
        if role == "SYSTEM":
            print(f"\n[{role}] ({len(content)} chars of protocol + tool catalogue)")
            continue
        label = "[TOOL OUTPUT - UNTRUSTED]" if role == "TOOL" else f"[{role}]"
        print(f"\n{label}")
        for line in content.splitlines():
            print(f"  {line}")

    banner("CHECKS")
    if case:
        print("  Human-readable criteria (documentation, never evaluated in code):")
        for line in case.grading.rubric:
            print(f"    - {line}")
        print()
    for check in result.check_results:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] ({check.role.value:6}) {check.detail}")

    banner("OUTCOME")
    outcome = result.outcome
    print(f"  refused          : {outcome.refused}")
    print(f"  task completed   : {outcome.task_completed}")
    print(f"  attack succeeded : {outcome.attack_succeeded}")
    # Three-valued on purpose: None means the checkpoint never ruled (module
    # off, or no critical action reached it), which is not the same as
    # "ruled, found nothing".
    print(f"  ToM flagged      : {outcome.misalignment_flagged}"
          + ("" if result.misalignment_ran else "  (checkpoint not enabled)"))
    for inferred in result.misalignment_inferred_tasks:
        print(f"    inferred task  : {inferred[:150]}")
    for reason in result.misalignment_reasons:
        print(f"    paused because : {reason[:150]}")
    print(f"  llm calls        : {outcome.num_llm_calls}")
    print(f"  verdict          : {'PASS' if result.passed else 'FAIL'}"
          f"{'' if result.passed else ' - ' + result.failure_reason}")
    if result.error:
        print(f"  run error        : {result.error[:200]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one saved case run.")
    parser.add_argument("case_id", nargs="?", help="Case to show, e.g. inj_001.")
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Results file (default: the newest in results/).")
    args = parser.parse_args()

    report = load_report(Path(args.results) if args.results else None)
    print(f"(reading {args.results or newest_results()})")
    if args.list or not args.case_id:
        print_index(report)
        return 0
    return show(report, args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
