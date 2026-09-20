"""Tests for the statistics behind every rate this project reports.

Measured 2026-09-20, before this module existed: not one of the frozen
ablation's headline results was statistically distinguishable from doing
nothing.

    ASR_inj  1/15 -> 0/15   Fisher one-sided p = 0.500
    HS       2/8  -> 0/8    p = 0.233
    passed   30/39 -> 31/39 p = 0.708

The table reported "0.07 -> 0.00" with no error bars, which reads as a result.
It is not one. A rate over 15 cases has a 95% interval roughly 0.30 wide, so
two such rates almost always overlap, and printing them bare invites a reader
(or a professor, or us) to believe a difference that the data does not carry.

Everything here is pure Python and exact - no scipy, no normal approximation
to a binomial with two successes.

No network calls in this file.
"""

from __future__ import annotations

import pytest

from src.eval.stats import (
    fisher_exact_one_sided,
    format_rate,
    required_n,
    wilson_interval,
)


class TestWilsonInterval:
    """Wilson rather than Wald: at k=0 or k=n, Wald gives a zero-width
    interval, which is exactly the misleading answer our HS=0.00 and
    ASR_inj=0.00 cells would otherwise print."""

    def test_zero_successes_still_has_width(self):
        low, high = wilson_interval(0, 15)
        assert low == 0.0
        assert high > 0.15, "0/15 does not mean the true rate is 0"

    def test_all_successes_still_has_width(self):
        low, high = wilson_interval(15, 15)
        assert high == 1.0
        assert low < 0.85

    def test_the_point_estimate_is_inside(self):
        for k, n in [(1, 15), (2, 8), (30, 39), (7, 10)]:
            low, high = wilson_interval(k, n)
            assert low <= k / n <= high

    def test_more_data_narrows_it(self):
        narrow = wilson_interval(10, 100)
        wide = wilson_interval(1, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_a_known_value(self):
        """1/15 at 95%: Wilson gives approximately [0.012, 0.298]."""
        low, high = wilson_interval(1, 15)
        assert low == pytest.approx(0.012, abs=0.005)
        assert high == pytest.approx(0.298, abs=0.005)

    def test_an_empty_denominator_is_the_whole_range(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestFisherExact:
    """Exact, because the cells are tiny. A chi-square on a table containing
    a 0 and a 2 is not trustworthy, and these tables always will."""

    def test_our_asr_result_is_not_significant(self):
        """The finding that motivated this module."""
        p = fisher_exact_one_sided(0, 15, 1, 15)
        assert p > 0.05
        assert p == pytest.approx(0.5, abs=0.01)

    def test_our_hs_result_is_not_significant(self):
        p = fisher_exact_one_sided(0, 8, 2, 8)
        assert p > 0.05

    def test_a_large_clear_effect_is_significant(self):
        p = fisher_exact_one_sided(0, 50, 25, 50)
        assert p < 0.001

    def test_identical_arms_are_maximally_unsurprising(self):
        assert fisher_exact_one_sided(5, 20, 5, 20) > 0.5

    def test_a_probability_is_returned(self):
        for args in [(0, 15, 1, 15), (3, 10, 7, 10), (0, 1, 1, 1)]:
            assert 0.0 <= fisher_exact_one_sided(*args) <= 1.0


class TestRequiredN:
    """How much data would we need? This is the number that decides whether
    the answer is 'write more cases' or 'write harder ones'."""

    def test_a_weak_effect_needs_a_lot(self):
        assert required_n(0.07) > 50

    def test_a_strong_effect_needs_little(self):
        assert required_n(0.40) < 15

    def test_stronger_attacks_beat_more_attacks(self):
        """The load-bearing result: the sample size we need is driven by the
        baseline rate. Raising ASR from 0.07 to 0.40 by using better payloads
        cuts the requirement by more than 5x - far cheaper than writing 5x the
        cases."""
        assert required_n(0.40) * 5 < required_n(0.07)

    def test_an_effect_of_zero_is_undetectable(self):
        assert required_n(0.0) is None


class TestFormatRate:
    """How a rate appears in every table we print."""

    def test_it_carries_the_interval(self):
        text = format_rate(1, 15)
        assert "1/15" in text
        assert "0.07" in text
        assert "[" in text and "]" in text

    def test_an_empty_denominator_says_so(self):
        assert format_rate(0, 0) == "n/a"

    def test_zero_is_not_presented_as_certainty(self):
        """0/15 must not render as a bare 0.00."""
        text = format_rate(0, 15)
        assert text != "0/15=0.00"
        assert "0.2" in text, "the upper bound belongs on screen"
