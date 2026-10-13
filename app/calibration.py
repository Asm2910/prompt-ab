"""Checking the LLM judge against a human: agreement statistics and test-set construction.

Pure functions only (no database, no API), so they are easy to test.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass

import numpy as np
from scipy import stats

CATEGORIES = ("billing", "technical", "account", "general")
SCORES = (1, 2, 3, 4, 5)
CORRUPTION_KINDS = ("wrong_category", "invented_summary", "unusable")
INVENTED_SUMMARIES = [
    "Customer wants to upgrade to the enterprise plan and asks for a call with sales.",
    "Customer reports a damaged shipment and wants a replacement sent.",
    "Customer asks how to export their data to a spreadsheet.",
    "Customer complains about the new logo and the website colours.",
    "Customer requests a change of the delivery address for an order.",
]
UNUSABLE_REPLY = "I'm sorry, I can't help with that request."


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Quadratic-weighted Cohen's kappa between two raters on the 1-5 scale.

    1 means perfect agreement and 0 means no better than chance (a judge that gives every item
    the same score gets exactly 0). Returns NaN only when BOTH raters give every item the same
    score, because then there is nothing to measure.
    """
    k = len(SCORES)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x - 1, y - 1] += 1
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return math.nan
    return float(1 - (weights * observed).sum() / denominator)


def bootstrap_ci(
    a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0
) -> tuple[float | None, float | None]:
    """95% bootstrap interval for the weighted kappa. Wide with few items: that is the point."""
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(a_arr), len(a_arr))
        kappa = weighted_kappa(list(a_arr[pick]), list(b_arr[pick]))
        if not math.isnan(kappa):
            values.append(kappa)
    if len(values) < n_boot // 2:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def confusion(human: list[int], judge: list[int]) -> list[list[int]]:
    """Rows are the human score (1-5), columns the judge score (1-5)."""
    table = [[0] * len(SCORES) for _ in SCORES]
    for h, j in zip(human, judge):
        table[h - 1][j - 1] += 1
    return table


def kappa_band(kappa: float) -> str:
    """A rough rule of thumb (Landis and Koch), not a standard."""
    if math.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "worse than chance"
    for limit, name in [(0.2, "slight, close to chance"), (0.4, "fair"),
                        (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= limit:
            return name
    return "almost perfect"


def agreement_report(human: list[int], judge: list[int]) -> dict:
    if len(human) != len(judge) or not human:
        raise ValueError("need two score lists of the same, non-zero length")
    n = len(human)
    kappa = weighted_kappa(human, judge)
    low, high = bootstrap_ci(human, judge)
    constant = len(set(human)) == 1 or len(set(judge)) == 1
    rho = math.nan if constant else float(stats.spearmanr(human, judge).statistic)
    return {
        "n": n,
        "exact_agreement": sum(h == j for h, j in zip(human, judge)) / n,
        "within_one": sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / n,
        "weighted_kappa": kappa,
        "kappa_ci_low": low,
        "kappa_ci_high": high,
        "kappa_band": kappa_band(kappa),
        "spearman": rho,
        "mean_bias": sum(j - h for h, j in zip(human, judge)) / n,  # judge minus human
        "confusion": confusion(human, judge),
    }


# ---------------------------------------------------------------------------
# Building the set of outputs to score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A real output from an experiment, with the ground-truth accuracy of its category."""

    request_id: int
    input_text: str
    output_text: str
    correct: bool


def parse_answer(output: str) -> dict:
    """The last JSON object with a category in a reply (chain-of-thought replies reason first)."""
    for block in reversed(re.findall(r"\{[^{}]*\}", output)):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("category"), str):
            return data
    raise ValueError("no answer object in the output")


def corrupt_output(output: str, kind: str, rng: random.Random) -> str:
    """Make a deliberately flawed version of a good output, so the test set has bad answers."""
    if kind == "unusable":
        return UNUSABLE_REPLY
    answer = parse_answer(output)
    category = answer["category"].strip().lower()
    summary = str(answer.get("summary", ""))
    if kind == "wrong_category":
        category = rng.choice([c for c in CATEGORIES if c != category])
    elif kind == "invented_summary":
        summary = rng.choice(INVENTED_SUMMARIES)
    else:
        raise ValueError(f"unknown corruption kind: {kind}")
    return json.dumps({"category": category, "summary": summary})


def build_items(
    candidates: list[Candidate],
    rng: random.Random,
    n_wrong: int = 8,
    n_right: int = 8,
    n_corrupt: int = 8,
) -> list[dict]:
    """A mixed set: real wrong answers, real right answers, and deliberately corrupted ones.

    Real wrong answers are the interesting cases. Corrupted ones guarantee some clearly bad
    outputs. The kind is recorded but never shown to the human scorer.
    """
    wrong = [c for c in candidates if not c.correct]
    right = [c for c in candidates if c.correct]
    rng.shuffle(wrong)
    rng.shuffle(right)

    items = []

    def add(kind: str, candidate: Candidate, output: str) -> None:
        items.append({
            "kind": kind,
            "request_id": candidate.request_id,
            "input": candidate.input_text,
            "output": output,
        })

    for c in wrong[:n_wrong]:
        add("real_wrong", c, c.output_text)
    for c in right[:n_right]:
        add("real_right", c, c.output_text)

    # Corrupt different requests from the ones shown as real right answers.
    made = 0
    for c in right[n_right:]:
        if made == n_corrupt:
            break
        kind = CORRUPTION_KINDS[made % len(CORRUPTION_KINDS)]
        try:
            add(f"corrupted_{kind}", c, corrupt_output(c.output_text, kind, rng))
        except ValueError:
            continue  # this output has no readable answer, so try the next one
        made += 1

    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item["id"] = f"cal-{i:03d}"  # assigned after shuffling, so the id reveals nothing
    return items