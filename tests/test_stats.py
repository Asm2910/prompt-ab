"""Tests for app/stats.py. Known answers, seeded simulations, and the no-peeking rule."""

import math

import numpy as np
import pytest
from scipy import stats as sps

from app.stats import (
    InsufficientDataError,
    compare_two,
    decide,
    required_sample_size,
)


def normal(seed: int, mean: float, std: float, n: int) -> list[float]:
    return list(np.random.default_rng(seed).normal(mean, std, n))


def test_welch_matches_hand_calculation():
    # diff = 1, each variance = 2.5, se = sqrt(2.5/5 + 2.5/5) = 1, df = 8
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]
    c = compare_two(a, b)
    margin = sps.t.ppf(0.975, 8)  # about 2.306
    assert c.diff == pytest.approx(1.0)
    assert (c.ci_low, c.ci_high) == (pytest.approx(1 - margin), pytest.approx(1 + margin))
    assert c.p_welch == pytest.approx(0.3466, abs=1e-3)


def test_p_values_match_scipy():
    a, b = normal(1, 0.0, 1.0, 60), normal(2, 0.4, 1.5, 80)
    c = compare_two(a, b)
    assert c.p_welch == pytest.approx(sps.ttest_ind(a, b, equal_var=False).pvalue)
    assert c.p_mann_whitney == pytest.approx(sps.mannwhitneyu(a, b).pvalue)


def test_large_true_effect_is_detected_and_ci_covers_it():
    c = compare_two(normal(3, 0.0, 1.0, 100), normal(4, 1.0, 1.0, 100))
    assert c.p_welch < 1e-6 and c.p_mann_whitney < 1e-6
    assert c.ci_low < 1.0 < c.ci_high
    assert c.cohens_d == pytest.approx(1.0, abs=0.3)


def test_identical_constant_groups_give_p_of_one():
    c = compare_two([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
    assert c.p_welch == 1.0 and c.p_mann_whitney == 1.0
    assert (c.ci_low, c.ci_high) == (0.0, 0.0)


def test_constant_but_different_groups_do_not_crash():
    c = compare_two([2.0] * 10, [5.0] * 10)
    assert c.diff == 3.0 and c.p_welch == 0.0
    assert c.p_mann_whitney < 0.001


def test_too_few_observations_raise():
    with pytest.raises(InsufficientDataError):
        compare_two([1.0], [1.0, 2.0])


def test_a_a_experiments_give_about_five_percent_false_positives():
    # No real difference exists, so about alpha of experiments should come out "significant",
    # and the 95% interval should cover the true difference (0) about 95% of the time.
    rng = np.random.default_rng(2024)
    trials, welch_hits, mw_hits, covered = 400, 0, 0, 0
    for _ in range(trials):
        c = compare_two(list(rng.normal(0, 1, 50)), list(rng.normal(0, 1, 50)))
        welch_hits += c.p_welch < 0.05
        mw_hits += c.p_mann_whitney < 0.05
        covered += c.ci_low <= 0 <= c.ci_high
    assert 0.02 <= welch_hits / trials <= 0.09
    assert 0.02 <= mw_hits / trials <= 0.09
    assert 0.92 <= covered / trials <= 0.98


def test_required_sample_size_matches_the_textbook_value():
    # 80% power, alpha 0.05, effect of half a standard deviation: about 63 per group.
    assert required_sample_size(std=1.0, effect=0.5) == 63
    assert required_sample_size(std=1.0, effect=0.25) > 3 * required_sample_size(std=1.0, effect=0.5)
    with pytest.raises(ValueError):
        required_sample_size(std=1.0, effect=0.0)


def test_mde_shrinks_as_the_sample_grows():
    small = compare_two(normal(5, 0, 1, 20), normal(6, 0, 1, 20))
    large = compare_two(normal(5, 0, 1, 200), normal(6, 0, 1, 200))
    assert large.mde < small.mde
    assert large.mde == pytest.approx(2.8 * math.sqrt(2 / 200), rel=0.3)


# ---------------------------------------------------------------------------
# decide(): the no-peeking rule and the verdicts
# ---------------------------------------------------------------------------


def two_groups(n_control, n_challenger, challenger_mean=1.0):
    return [
        (1, "control", normal(10, 0.0, 1.0, n_control)),
        (2, "challenger", normal(11, challenger_mean, 1.0, n_challenger)),
    ]


def test_no_decision_before_the_target_sample_size():
    d = decide(two_groups(60, 100), target_n=100)
    assert d.status == "not_ready" and d.interim and d.winner_label is None
    assert d.comparisons[0].verdict == "pending"
    # The interim numbers are still shown, but they are not a verdict.
    assert d.comparisons[0].p_decision is not None
    assert [v.samples_needed for v in d.variants] == [40, 0]


def test_winner_is_declared_once_every_variant_reaches_the_target():
    d = decide(two_groups(100, 100), target_n=100)
    assert d.status == "winner" and not d.interim
    assert d.winner_label == "challenger" and d.comparisons[0].verdict == "better"


def test_lower_is_better_metrics_pick_the_lower_variant():
    groups = [(1, "slow", normal(1, 2.0, 0.5, 100)), (2, "fast", normal(2, 1.0, 0.5, 100))]
    assert decide(groups, 100, higher_is_better=False).winner_label == "fast"
    assert decide(groups, 100, higher_is_better=True).status == "control_holds"


def test_control_holds_when_a_challenger_is_significantly_worse():
    d = decide(two_groups(100, 100, challenger_mean=-1.0), target_n=100)
    assert d.status == "control_holds" and d.winner_label == "control"
    assert d.comparisons[0].verdict == "worse"


def test_no_difference_when_the_variants_are_the_same():
    groups = [(1, "a", normal(7, 0, 1, 100)), (2, "b", normal(8, 0, 1, 100))]
    d = decide(groups, target_n=100)
    assert d.status == "no_difference" and d.winner_label is None
    assert d.comparisons[0].verdict == "no_difference"


def test_bonferroni_adjustment_scales_with_the_number_of_challengers():
    groups = [
        (1, "control", normal(20, 0.0, 1.0, 100)),
        (2, "b", normal(21, 0.3, 1.0, 100)),
        (3, "c", normal(22, 0.0, 1.0, 100)),
    ]
    d = decide(groups, target_n=100, test="welch")
    for r in d.comparisons:
        assert r.p_adjusted == pytest.approx(min(1.0, r.p_decision * 2))


def test_decision_uses_the_chosen_test():
    groups = two_groups(100, 100, challenger_mean=0.2)
    welch = decide(groups, 100, test="welch").comparisons[0]
    ranks = decide(groups, 100, test="mann_whitney").comparisons[0]
    assert welch.p_decision == welch.p_welch
    assert ranks.p_decision == ranks.p_mann_whitney


def test_one_observation_per_group_cannot_decide():
    groups = [(1, "a", [1.0]), (2, "b", [2.0])]
    d = decide(groups, target_n=1)
    assert d.status == "not_ready"
    assert d.comparisons[0].verdict == "insufficient_data"
    assert d.variants[0].samples_needed == 1  # a test needs at least 2 observations


def test_empty_groups_do_not_crash():
    d = decide([(1, "a", []), (2, "b", [])], target_n=10)
    assert d.status == "not_ready" and d.variants[0].mean is None