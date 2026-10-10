"""Statistics for comparing prompt variants. Pure functions: no database, no I/O."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

from scipy import stats as sps

Test = Literal["welch", "mann_whitney"]


class InsufficientDataError(ValueError):
    """A group has too few observations for the requested calculation."""


@dataclass(frozen=True)
class Comparison:
    """Group B compared with group A. `diff` is mean(B) - mean(A)."""

    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_welch: float
    p_mann_whitney: float
    cohens_d: float | None
    mde: float  # smallest difference this sample size could reliably detect


def _mann_whitney_p(a: list[float], b: list[float]) -> float:
    # If every value is identical the statistic has zero variance (0/0), so report p = 1.
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def compare_two(
    a: list[float], b: list[float], alpha: float = 0.05, power: float = 0.8
) -> Comparison:
    """Welch t-test, Mann-Whitney U, a confidence interval for the mean difference, and the MDE."""
    if len(a) < 2 or len(b) < 2:
        raise InsufficientDataError("each group needs at least 2 observations")
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    diff = mean_b - mean_a
    se = math.sqrt(var_a / n_a + var_b / n_b)

    if se == 0:
        # Both groups are constant, so there is no spread to test against.
        ci_low = ci_high = diff
        p_welch = 1.0 if diff == 0 else 0.0
    else:
        # Welch-Satterthwaite degrees of freedom: valid when the two variances differ.
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        margin = float(sps.t.ppf(1 - alpha / 2, df)) * se
        ci_low, ci_high = diff - margin, diff + margin
        p_welch = float(2 * sps.t.sf(abs(diff / se), df))

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    cohens_d = diff / math.sqrt(pooled_var) if pooled_var > 0 else None
    z = float(sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power))
    return Comparison(
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=1 - alpha,
        p_welch=p_welch,
        p_mann_whitney=_mann_whitney_p(a, b),
        cohens_d=cohens_d,
        mde=z * se,
    )


def required_sample_size(
    std: float, effect: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Observations needed PER VARIANT to detect a difference of `effect` (normal approximation)."""
    if std <= 0 or effect <= 0:
        raise ValueError("std and effect must be greater than 0")
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ValueError("alpha and power must be between 0 and 1")
    z = float(sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power))
    return math.ceil(2 * (z * std / effect) ** 2)


@dataclass(frozen=True)
class VariantStats:
    variant_id: int
    label: str
    n: int
    mean: float | None
    std: float | None
    samples_needed: int  # extra observations before a decision is allowed


@dataclass(frozen=True)
class ChallengerResult:
    """One challenger against the control (the first variant)."""

    variant_id: int
    label: str
    verdict: str  # pending | better | worse | no_difference | insufficient_data
    diff: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float | None
    p_welch: float | None
    p_mann_whitney: float | None
    p_decision: float | None  # the p-value from the pre-chosen test
    p_adjusted: float | None  # Bonferroni: p_decision times the number of challengers
    cohens_d: float | None
    mde: float | None


@dataclass(frozen=True)
class Decision:
    status: str  # not_ready | winner | control_holds | no_difference
    winner_id: int | None
    winner_label: str | None
    interim: bool
    variants: list[VariantStats]
    comparisons: list[ChallengerResult]


def decide(
    groups: list[tuple[int, str, list[float]]],
    target_n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    higher_is_better: bool = True,
    test: Test = "mann_whitney",
) -> Decision:
    """Compare every challenger with the control (groups[0]) and apply the no-peeking rule.

    Nothing is declared until EVERY variant has `target_n` observations. Before that the
    numbers are returned for display, but every verdict is 'pending'.
    """
    if len(groups) < 2:
        raise ValueError("need a control and at least one challenger")
    needed = max(target_n, 2)  # a test needs at least two observations per group
    _, control_label, control_values = groups[0]
    control_id = groups[0][0]
    challengers = groups[1:]
    k = len(challengers)
    ready = all(len(values) >= needed for _, _, values in groups)

    variants = [
        VariantStats(
            variant_id=vid,
            label=label,
            n=len(values),
            mean=statistics.fmean(values) if values else None,
            std=statistics.stdev(values) if len(values) > 1 else None,
            samples_needed=max(0, needed - len(values)),
        )
        for vid, label, values in groups
    ]
    stats_by_id = {v.variant_id: v for v in variants}

    results: list[ChallengerResult] = []
    for vid, label, values in challengers:
        if len(control_values) < 2 or len(values) < 2:
            results.append(
                ChallengerResult(vid, label, "insufficient_data", *([None] * 10))
            )
            continue
        c = compare_two(control_values, values, alpha, power)
        p = c.p_welch if test == "welch" else c.p_mann_whitney
        p_adjusted = min(1.0, p * k)
        if not ready:
            verdict = "pending"
        elif p_adjusted >= alpha or c.diff == 0:
            verdict = "no_difference"
        else:
            # Direction comes from the mean difference.
            improved = c.diff > 0 if higher_is_better else c.diff < 0
            verdict = "better" if improved else "worse"
        results.append(
            ChallengerResult(
                variant_id=vid,
                label=label,
                verdict=verdict,
                diff=c.diff,
                ci_low=c.ci_low,
                ci_high=c.ci_high,
                ci_level=c.ci_level,
                p_welch=c.p_welch,
                p_mann_whitney=c.p_mann_whitney,
                p_decision=p,
                p_adjusted=p_adjusted,
                cohens_d=c.cohens_d,
                mde=c.mde,
            )
        )

    if not ready:
        return Decision("not_ready", None, None, True, variants, results)
    better = [r for r in results if r.verdict == "better"]
    if better:
        pick = max if higher_is_better else min
        best = pick(better, key=lambda r: stats_by_id[r.variant_id].mean)
        return Decision("winner", best.variant_id, best.label, False, variants, results)
    if any(r.verdict == "worse" for r in results):
        return Decision("control_holds", control_id, control_label, False, variants, results)
    return Decision("no_difference", None, None, False, variants, results)