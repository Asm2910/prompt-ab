"""Tests for the deterministic traffic splitter."""

import pytest

from app.splitter import assign_variant

SPLIT = [(10, 0.5), (20, 0.5)]


def test_same_user_always_gets_same_variant():
    first = assign_variant(1, "user-42", SPLIT)
    assert all(assign_variant(1, "user-42", SPLIT) == first for _ in range(100))


def test_different_experiments_assign_independently():
    # Salting with the experiment id: the same 200 users must not map identically.
    in_exp_1 = [assign_variant(1, f"u{i}", SPLIT) for i in range(200)]
    in_exp_2 = [assign_variant(2, f"u{i}", SPLIT) for i in range(200)]
    assert in_exp_1 != in_exp_2


@pytest.mark.parametrize(
    "variants",
    [
        [(1, 0.5), (2, 0.5)],
        [(1, 0.8), (2, 0.1), (3, 0.1)],
        [(1, 0.34), (2, 0.33), (3, 0.33)],
    ],
)
def test_split_matches_configured_shares(variants):
    n = 10_000
    counts = {vid: 0 for vid, _ in variants}
    for i in range(n):
        counts[assign_variant(7, f"user-{i}", variants)] += 1
    for vid, share in variants:
        assert counts[vid] / n == pytest.approx(share, abs=0.02)


def test_rejects_empty_variant_list():
    with pytest.raises(ValueError):
        assign_variant(1, "u", [])


def test_float_rounding_never_returns_nothing():
    # Ten shares of 0.1 sum to slightly under 1.0; every user must still get a variant.
    variants = [(i, 0.1) for i in range(10)]
    results = {assign_variant(3, f"u{i}", variants) for i in range(2000)}
    assert results <= set(range(10))