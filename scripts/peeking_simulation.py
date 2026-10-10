"""Why peeking breaks significance tests.

Simulates A/A experiments (both variants identical, so any "winner" is a false alarm) and
compares two ways of reading them:
  fixed horizon: test once, when the planned sample size is reached
  peeking:       test every few samples and stop at the first p < alpha

Run:  uv run python scripts/peeking_simulation.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def main(
    n_experiments: int = 2000,
    max_n: int = 200,
    look_every: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n_experiments, max_n))
    b = rng.normal(0, 1, (n_experiments, max_n))  # same distribution as a: no real effect

    looks = list(range(look_every, max_n + 1, look_every))
    # One Welch t-test per look, run across all experiments at once.
    p_by_look = np.array(
        [stats.ttest_ind(a[:, :n], b[:, :n], axis=1, equal_var=False).pvalue for n in looks]
    )
    fixed = (p_by_look[-1] < alpha).mean()
    peeking = (p_by_look < alpha).any(axis=0).mean()

    print(f"{n_experiments} A/A experiments, {max_n} samples per variant, alpha = {alpha}")
    print(f"Fixed horizon (one look at n={max_n}):      {fixed:.1%} false winners")
    print(f"Peeking ({len(looks)} looks, stop at first hit): {peeking:.1%} false winners")


if __name__ == "__main__":
    main()