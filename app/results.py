"""Turns an experiment's stored metrics into a statistical verdict."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.experiments import get_experiment, get_variants
from app.metrics import BUILTIN_METRICS, metric_values
from app.stats import ChallengerResult, Decision, Test, VariantStats, decide

METRIC_ALIASES = {"quality": "judge_quality"}
HIGHER_IS_BETTER = {"judge_quality", "json_valid", "json_bare"}
LOWER_IS_BETTER = set(BUILTIN_METRICS)  # latency_ms, input_tokens, output_tokens, cost_usd
# Judge scores are ordinal (1-5) and latency is skewed, so the rank-based test decides for
# those two. The others use Welch's t-test. Both p-values are always reported.
RANK_TEST_METRICS = {"judge_quality", "latency_ms"}


class UnknownMetricError(ValueError):
    pass


def resolve_metric(name: str) -> str:
    resolved = METRIC_ALIASES.get(name, name)
    known = HIGHER_IS_BETTER | LOWER_IS_BETTER
    if resolved not in known:
        options = ", ".join(sorted(known | set(METRIC_ALIASES)))
        raise UnknownMetricError(f"unknown metric {name!r}; choose one of: {options}")
    return resolved


@dataclass(frozen=True)
class ExperimentResults:
    experiment_id: int
    experiment_status: str
    metric: str
    higher_is_better: bool
    decision_test: str
    alpha: float
    power: float
    target_per_variant: int
    decision: str
    winner: str | None
    interim: bool
    variants: list[VariantStats]
    comparisons: list[ChallengerResult]
    note: str


def _note(decision: Decision, metric: str, alpha: float) -> str:
    if decision.status == "not_ready":
        short = max(v.samples_needed for v in decision.variants)
        return (
            f"Interim numbers only, not a decision. The variant with the fewest samples "
            f"needs {short} more {metric} observations before a verdict is allowed."
        )
    if decision.status == "winner":
        return (
            f"{decision.winner_label} beat the control on {metric} (adjusted p below {alpha}). "
            "Stop the experiment and read this result once: checking again after more "
            "data arrives is peeking and inflates false positives."
        )
    if decision.status == "control_holds":
        return (
            "No challenger beat the control, and at least one is significantly worse. "
            "Keep the control."
        )
    return (
        "No significant difference at the planned sample size. That is a result: keep "
        "the control, or test a bigger change."
    )


def analyze_experiment(
    session: Session,
    experiment_id: int,
    alpha: float = 0.05,
    power: float = 0.8,
    metric: str | None = None,
) -> ExperimentResults:
    """Compare every variant with the control (the first variant listed at creation)."""
    experiment = get_experiment(session, experiment_id)
    metric_name = resolve_metric(metric or experiment.primary_metric)
    higher_is_better = metric_name in HIGHER_IS_BETTER
    test: Test = "mann_whitney" if metric_name in RANK_TEST_METRICS else "welch"

    values = metric_values(session, experiment_id, metric_name)
    groups = [
        (v.id, v.label, values.get(v.id, [])) for v in get_variants(session, experiment_id)
    ]
    decision = decide(
        groups, experiment.target_sample_size, alpha, power, higher_is_better, test
    )
    return ExperimentResults(
        experiment_id=experiment.id,
        experiment_status=experiment.status,
        metric=metric_name,
        higher_is_better=higher_is_better,
        decision_test=test,
        alpha=alpha,
        power=power,
        target_per_variant=experiment.target_sample_size,
        decision=decision.status,
        winner=decision.winner_label,
        interim=decision.interim,
        variants=decision.variants,
        comparisons=decision.comparisons,
        note=_note(decision, metric_name, alpha),
    )