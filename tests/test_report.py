"""The Phase 6 report generator (CLAUDE.md §10, §9.2).

The report is the deliverable, so the things worth testing are the ways it
could be *dishonest* rather than the ways it could be ugly: a row built from
the wrong backbone, a missing term omitted rather than named, the §9.2 caveat
quietly paraphrased into a comparison the project cannot support.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config import settings
from src.eval import report as report_module
from src.eval.schemas import Outcome, RunResult, SuiteReport
from tests.test_gai import run


def snapshot(stem: str, directory, results, model: str | None = None) -> None:
    """Write one saved run file the report will find."""
    SuiteReport(
        condition="X",
        suites=sorted({r.suite for r in results}),
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(),
        backbone_model=model if model is not None else settings.BACKBONE_MODEL,
    ).write(directory / f"{stem}.json")


def a_suite(attack_rate: bool, complied: bool) -> list[RunResult]:
    """A minimal but complete case set: injection + harmful + benign."""
    return [
        run("inj_001", suite="injection", attack=attack_rate,
            completed=not attack_rate),
        run("inj_002", suite="injection", attack=False, completed=True),
        run("harm_001", suite="direct_harm", expects_refusal=True,
            refusal_acceptable=True, refused=not complied, completed=complied,
            expects_task_completion=False),
        run("harm_002", suite="direct_harm", expects_refusal=True,
            refusal_acceptable=True, refused=True, completed=False,
            expects_task_completion=False),
        run("benign_001", suite="direct_harm", completed=True),
    ]


@pytest.fixture
def built(tmp_path):
    """A results directory with all five ablation rows, and the report."""
    for index, (label, stem) in enumerate(report_module.ABLATION_ROWS):
        snapshot(stem, tmp_path, a_suite(attack_rate=index == 0,
                                         complied=index == 0))
    path = report_module.build_report(results_dir=tmp_path)
    return path, path.read_text(encoding="utf-8")


class TestStructure:
    def test_writes_report_md(self, built):
        path, _ = built
        assert path.name == "report.md"
        assert path.exists()

    @pytest.mark.parametrize("heading", [
        "## 1. Headline",
        "## 2. Sub-metrics",
        "## 3. Is any of it real?",
        "## 4. Weight sensitivity",
        "## 5. Cumulative ablation",
        "## 6. Single-module isolation",
        "## 7. LAT",
        "## 8. Terms the index does not contain",
        "## 9. The source papers",
        "## 10. Limitations",
        "## 11. Provenance",
    ])
    def test_every_section_required_by_phase_6_is_present(self, built, heading):
        _, text = built
        assert heading in text

    def test_names_the_pinned_backbone(self, built):
        _, text = built
        assert settings.BACKBONE_MODEL in text

    def test_every_weight_vector_appears(self, built):
        _, text = built
        for label in settings.GAI_WEIGHT_VECTORS:
            assert label in text

    def test_refuses_to_build_with_no_snapshots(self, tmp_path):
        with pytest.raises(SystemExit, match="No ablation snapshots"):
            report_module.build_report(results_dir=tmp_path)


class TestReferenceNumbers:
    """§9.2: context for scale, never a comparison. The caveat is what keeps
    that distinction load-bearing, so it must survive verbatim."""

    def test_the_caveat_appears_verbatim(self, built):
        _, text = built
        assert report_module.REFERENCE_CAVEAT in text

    def test_the_caveat_still_says_the_comparison_is_not_statistical(self):
        caveat = report_module.REFERENCE_CAVEAT
        assert "not as a benchmark" in caveat
        assert "statistically compared against" in caveat
        assert "Condition B vs. Condition A" in caveat

    def test_every_published_number_is_listed(self, built):
        _, text = built
        for paper, _, number in report_module.REFERENCE_NUMBERS:
            assert paper in text
            assert number in text

    def test_all_five_measured_papers_are_represented(self):
        papers = {p for p, _, _ in report_module.REFERENCE_NUMBERS}
        assert {"IPIGuard", "ShieldMCP", "AgentHarm", "InferAct",
                "AgentVigil", "SIRAJ"} <= papers


class TestHonesty:
    def test_a_snapshot_from_another_backbone_is_refused(self, tmp_path):
        """The handoff's bug: a gemini file once supplied the Condition A row
        of a table whose other four rows were qwen."""
        snapshot("frozen_1_condition_a", tmp_path, a_suite(True, True),
                 model="some-other/model")
        assert report_module.load_snapshot(
            "Condition A", "frozen_1_condition_a", tmp_path
        ) is None

    def test_a_snapshot_on_the_pinned_backbone_loads(self, tmp_path):
        snapshot("frozen_1_condition_a", tmp_path, a_suite(True, True))
        assert report_module.load_snapshot(
            "Condition A", "frozen_1_condition_a", tmp_path
        ) is not None

    def test_unbuilt_terms_are_named_not_omitted(self, built):
        """§9 forbids silently dropping a term from the index."""
        _, text = built
        assert "DIV_ASR" in text
        assert "never built" in text
        assert "renormalis" in text

    def test_says_n_equals_one_and_why(self, built):
        _, text = built
        assert "N=1" in text or "N = 1" in text
        assert "requests/minute" in text

    def test_isolation_gap_is_stated_when_not_run(self, built):
        _, text = built
        assert "Not run" in text

    def test_lat_absence_is_stated_when_not_timed(self, built):
        _, text = built
        assert "Not measured" in text

    def test_every_row_names_its_source_file(self, built):
        _, text = built
        for _, stem in report_module.ABLATION_ROWS:
            assert f"`{stem}.json`" in text

    def test_reports_when_case_sets_differ(self, tmp_path):
        """A trend across rows measured over different cases is partly a
        change of test, and the report must say so rather than draw it."""
        for index, (_, stem) in enumerate(report_module.ABLATION_ROWS):
            results = a_suite(False, False)
            if index == len(report_module.ABLATION_ROWS) - 1:
                results.append(run("extra_001", suite="injection", attack=False))
            snapshot(stem, tmp_path, results)
        text = report_module.build_report(results_dir=tmp_path).read_text(
            encoding="utf-8"
        )
        assert "Case sets differ" in text

    def test_confirms_comparability_when_case_sets_match(self, built):
        _, text = built
        assert "directly comparable" in text


class TestBarChart:
    def test_renders_one_row_per_configuration(self):
        chart = report_module._bar_chart(
            [("A", 0.90), ("B", 0.95), ("C", 0.80)]
        )
        for label in ("A", "B", "C"):
            assert label in chart
        assert "0.950" in chart

    def test_states_that_bars_are_not_scaled_to_zero_one(self):
        """Over these results the whole spread is ~0.07 wide, so the axis is
        the observed range - and a reader has to be told that."""
        chart = report_module._bar_chart([("A", 0.90), ("B", 0.95)])
        assert "not to 0-1" in chart

    def test_the_largest_value_gets_the_longest_bar(self):
        rows = report_module._bar_chart([("A", 0.10), ("B", 0.90)]).splitlines()
        assert rows[0].count("#") < rows[1].count("#")

    def test_handles_a_single_row_without_dividing_by_zero(self):
        assert "0.500" in report_module._bar_chart([("only", 0.5)])


class TestRefusalAcceptableIds:
    def test_reads_three_valued_expectations_from_the_live_suite(self):
        ids = report_module.refusal_acceptable_ids()
        from src.eval.schemas import load_suites

        expected = {
            c.id for c in load_suites(settings.TESTSUITES_DIR)
            if c.expects.should_refuse is not False
        }
        assert ids == expected
        assert ids, "no case accepts a refusal; the correction would be a no-op"
