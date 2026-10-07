"""Deterministic traffic splitting: the same user always lands in the same variant."""

from __future__ import annotations

import hashlib


def assign_variant(
    experiment_id: int, user_id: str, variants: list[tuple[int, float]]
) -> int:
    """Pick a variant id for this user. `variants` is [(variant_id, share), ...], shares sum to 1.

    The hash is salted with the experiment id, so one user can land in different
    arms of different experiments. Keep the variant order stable (we sort by id).
    """
    if not variants:
        raise ValueError("at least one variant is required")
    key = f"{experiment_id}:{user_id}".encode()
    # Map the hash to a stable number in [0, 1), then walk the cumulative shares.
    bucket = (int(hashlib.sha256(key).hexdigest(), 16) % 10_000) / 10_000
    cumulative = 0.0
    for variant_id, share in variants:
        cumulative += share
        if bucket < cumulative:
            return variant_id
    return variants[-1][0]  # guards against float rounding at the very top