"""Confidence intervals and significance tests for the rates we report.

WHY THIS EXISTS
---------------
Measured on 2026-09-20, over the frozen 39-case ablation: **not one of this
project's headline results was statistically distinguishable from doing
nothing.**

    ASR_inj   1/15 = 0.07  ->  0/15 = 0.00     Fisher one-sided p = 0.500
    HS        2/8  = 0.25  ->  0/8  = 0.00     p = 0.233
    passed   30/39          -> 31/39           p = 0.708

The ablation table printed those as `0.07` and `0.00`, bare. That reads as a
result. It is not one: a rate over 15 cases carries a 95% interval about 0.30
wide, so the two arms overlap almost completely. Reporting the point estimate
alone invites the reader - and us - to believe a difference the data cannot
support, which is the same class of error as the silent-number bugs in
docs/HANDOFF.md §1, just committed in the presentation layer instead of the
code.

So every rate this project prints now carries its interval.

WHY THESE PARTICULAR METHODS
----------------------------
**Wilson, not Wald.** Wald (`p ± z·sqrt(p(1-p)/n)`) collapses to zero width at
`k=0` and `k=n`. Our best cells are exactly those - `HS = 0/8`,
`ASR_inj = 0/15` - so Wald would report our strongest claims as certainties.
Wilson stays sensible at the boundaries, which is the whole reason to prefer
it here.

**Fisher exact, not chi-square.** The tables contain cells of 0 and 2. A
chi-square approximation is not trustworthy at those counts, and the exact
hypergeometric sum is a dozen lines.

Pure Python on purpose: `scipy` is a large dependency to add to a project
whose requirements.txt is five lines, and nothing here needs more than
`math.comb`.
"""

from __future__ import annotations

from math import comb, sqrt

# 95% two-sided normal quantile. The only constant we would have imported
# scipy for.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns `(low, high)`, clamped to [0, 1]. An empty denominator returns the
    whole range - with no data, every rate is possible, and that is the honest
    answer rather than a crash or a zero.
    """
    if total <= 0:
        return (0.0, 1.0)

    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = (
        z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def fisher_exact_one_sided(
    a_successes: int, a_total: int, b_successes: int, b_total: int
) -> float:
    """One-sided Fisher exact test: is arm A's rate *lower* than arm B's?

    Argument order is "defended arm first", because that is the question this
    project always asks - did the defense reduce the rate. Returns the
    probability of seeing a table at least this extreme under the null
    hypothesis that both arms share one rate.
    """
    a_failures = a_total - a_successes
    b_failures = b_total - b_successes

    n = a_total + b_total
    row_total = a_total
    col_total = a_successes + b_successes
    if n == 0 or col_total == 0 or col_total == n:
        # No variation to explain: every arrangement is equally unsurprising.
        return 1.0

    total = 0.0
    for x in range(0, min(row_total, col_total) + 1):
        if row_total - x > n - col_total:
            continue
        probability = (
            comb(col_total, x)
            * comb(n - col_total, row_total - x)
            / comb(n, row_total)
        )
        if x <= a_successes:
            total += probability
    del a_failures, b_failures  # kept above for readability of the 2x2
    return min(1.0, total)


def required_n(
    baseline_rate: float,
    defended_rate: float = 0.0,
    alpha: float = 0.05,
    cap: int = 4000,
) -> int | None:
    """Cases per arm needed for `baseline -> defended` to reach significance.

    Assumes the observed effect is the true one, and asks the smallest `n` at
    which the *expected* table clears `alpha`. A planning tool, not a claim.

    The reason it earns a place in the codebase: the answer is driven by
    `baseline_rate`, not by how many cases we are willing to write. Measured -
    a baseline of 0.07 needs n=65, a baseline of 0.40 needs n=9. Our injection
    suite sits at 0.07 because the payloads mostly bounce off the backbone, so
    **writing harder attacks is more than five times cheaper than writing more
    of them.** That is what pointed at AgentDojo's templates rather than at a
    bigger hand-written corpus.

    Returns None when no achievable `n` helps (a zero effect, or beyond `cap`).
    """
    if baseline_rate <= defended_rate:
        return None

    for n in range(4, cap):
        a = round(defended_rate * n)
        b = round(baseline_rate * n)
        if b == 0:
            continue
        if fisher_exact_one_sided(a, n, b, n) < alpha:
            return n
    return None


def format_rate(successes: int, total: int, decimals: int = 2) -> str:
    """One rate, with its interval, as it appears in every table we print.

    `0/15` renders as `0/15=0.00 [0.00-0.20]` rather than `0/15=0.00`, because
    the upper bound is the part a reader needs in order not to over-read a
    perfect-looking cell.
    """
    if total <= 0:
        return "n/a"
    low, high = wilson_interval(successes, total)
    return (
        f"{successes}/{total}={successes / total:.{decimals}f} "
        f"[{low:.{decimals}f}-{high:.{decimals}f}]"
    )


def compare(
    label: str,
    a_successes: int, a_total: int,
    b_successes: int, b_total: int,
    lower_is_better: bool = True,
) -> str:
    """A full A/B line: both rates with intervals, and whether it is real.

    `lower_is_better` picks which direction counts as improvement, so the same
    function serves ASR_inj (down is good) and MF1-style rates (up is good).
    """
    if lower_is_better:
        p = fisher_exact_one_sided(b_successes, b_total, a_successes, a_total)
    else:
        p = fisher_exact_one_sided(a_successes, a_total, b_successes, b_total)
    verdict = "significant" if p < 0.05 else "NOT significant"
    return (
        f"{label}: {format_rate(a_successes, a_total)} -> "
        f"{format_rate(b_successes, b_total)}  (p={p:.3f}, {verdict})"
    )


__all__ = [
    "Z_95",
    "compare",
    "fisher_exact_one_sided",
    "format_rate",
    "required_n",
    "wilson_interval",
]
