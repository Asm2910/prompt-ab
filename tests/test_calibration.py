"""Tests for app/calibration.py: agreement statistics and test-set construction."""

import json
import math
import random

import pytest

from app.calibration import (
    CATEGORIES,
    Candidate,
    agreement_report,
    bootstrap_ci,
    build_items,
    corrupt_output,
    kappa_band,
    weighted_kappa,
)

GOOD = '```json\n{"category": "billing", "summary": "Customer was charged twice."}\n```'
COT = 'The customer mentions a double charge first.\n{"category": "billing", "summary": "Double charge."}'


def test_perfect_agreement_gives_kappa_one():
    scores = [1, 2, 3, 4, 5, 3, 4, 2]
    assert weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_kappa_matches_the_scikit_learn_value():
    # Value computed with sklearn.metrics.cohen_kappa_score(..., weights="quadratic").
    a = [1, 2, 3, 4, 5, 3, 2, 4, 5, 1, 3, 4]
    b = [1, 2, 3, 5, 4, 3, 3, 4, 5, 2, 3, 3]
    assert weighted_kappa(a, b) == pytest.approx(0.8636363636, abs=1e-9)


def test_a_rater_who_never_varies_scores_zero_and_two_constant_raters_are_undefined():
    assert weighted_kappa([1, 3, 5, 4], [5, 5, 5, 5]) == pytest.approx(0.0)
    assert math.isnan(weighted_kappa([5, 5, 5], [5, 5, 5]))


def test_kappa_penalises_big_disagreements_more_than_small_ones():
    human = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    off_by_one = [2, 3, 4, 5, 4, 2, 3, 4, 5, 4]
    off_by_four = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    assert weighted_kappa(human, off_by_one) > weighted_kappa(human, off_by_four)
    assert weighted_kappa(human, off_by_four) < 0


def test_bootstrap_interval_brackets_the_estimate_and_is_wide_for_small_samples():
    human = [1, 2, 3, 4, 5, 3, 2, 4, 5, 1, 3, 4]
    judge = [1, 2, 3, 5, 4, 3, 3, 4, 5, 2, 3, 3]
    low, high = bootstrap_ci(human, judge)
    assert low < weighted_kappa(human, judge) < high
    assert high - low > 0.1  # twelve items cannot pin kappa down tightly


def test_agreement_report_numbers():
    human = [5, 5, 4, 2, 1, 3]
    judge = [5, 4, 4, 3, 1, 5]
    report = agreement_report(human, judge)
    assert report["n"] == 6
    assert report["exact_agreement"] == pytest.approx(3 / 6)
    assert report["within_one"] == pytest.approx(5 / 6)
    assert report["mean_bias"] == pytest.approx((0 - 1 + 0 + 1 + 0 + 2) / 6)  # judge minus human
    assert report["confusion"][0][0] == 1 and report["confusion"][4][3] == 1
    assert 0 < report["spearman"] <= 1


def test_agreement_report_handles_a_judge_that_gives_everything_the_same_score():
    report = agreement_report([1, 3, 5, 4], [5, 5, 5, 5])
    assert report["weighted_kappa"] == pytest.approx(0.0)  # no better than chance
    assert math.isnan(report["spearman"])  # a correlation needs both raters to vary
    assert report["kappa_band"] == "slight, close to chance"
    assert report["mean_bias"] == pytest.approx((4 + 2 + 0 + 1) / 4)


def test_kappa_bands():
    assert kappa_band(0.9) == "almost perfect" and kappa_band(0.7) == "substantial"
    assert kappa_band(0.5) == "moderate" and kappa_band(-0.1) == "worse than chance"
    assert kappa_band(0.0) == "slight, close to chance"


def test_corruptions_change_exactly_what_they_claim_to():
    rng = random.Random(1)
    wrong = json.loads(corrupt_output(GOOD, "wrong_category", rng))
    assert wrong["category"] in CATEGORIES and wrong["category"] != "billing"
    assert wrong["summary"] == "Customer was charged twice."

    invented = json.loads(corrupt_output(COT, "invented_summary", rng))
    assert invented["category"] == "billing" and invented["summary"] != "Double charge."

    assert "{" not in corrupt_output(GOOD, "unusable", rng)
    with pytest.raises(ValueError):
        corrupt_output(GOOD, "mystery", rng)
    with pytest.raises(ValueError):
        corrupt_output("no answer here", "wrong_category", rng)


def candidates(n_right=20, n_wrong=12):
    out = [Candidate(i, f"email {i}", GOOD, True) for i in range(n_right)]
    out += [Candidate(100 + i, f"email {100 + i}", COT, False) for i in range(n_wrong)]
    return out


def test_build_items_mixes_real_wrong_real_right_and_corrupted():
    items = build_items(candidates(), random.Random(3))
    kinds = [item["kind"] for item in items]
    assert kinds.count("real_wrong") == 8 and kinds.count("real_right") == 8
    assert sum(k.startswith("corrupted_") for k in kinds) == 8
    assert {k for k in kinds if k.startswith("corrupted_")} == {
        "corrupted_wrong_category", "corrupted_invented_summary", "corrupted_unusable"
    }


def test_build_items_ids_hide_the_kind_and_requests_are_not_reused():
    items = build_items(candidates(), random.Random(3))
    assert sorted(item["id"] for item in items) == [f"cal-{i:03d}" for i in range(1, 25)]
    # After shuffling, ids carry no information about the kind: the first id is not always real.
    assert len({item["kind"] for item in items[:6]}) > 1
    # The same request is never shown twice (a real right answer and its own corruption).
    assert len({item["request_id"] for item in items}) == len(items)
    # Real items carry the output unchanged. Corrupted ones were changed.
    for item in items:
        if item["kind"].startswith("real_"):
            assert item["output"] in (GOOD, COT)
        else:
            assert item["output"] != GOOD


def test_build_items_is_repeatable_and_different_per_seed():
    first = build_items(candidates(), random.Random(3))
    again = build_items(candidates(), random.Random(3))
    other = build_items(candidates(), random.Random(4))
    assert first == again
    assert [i["request_id"] for i in first] != [i["request_id"] for i in other]


def test_build_items_copes_with_too_few_wrong_answers():
    items = build_items(candidates(n_right=30, n_wrong=2), random.Random(1))
    assert [i["kind"] for i in items].count("real_wrong") == 2