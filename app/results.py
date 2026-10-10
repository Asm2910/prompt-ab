"""Turns an experiment's stored metrics into a statistical verdict."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.experiments import get_experiment, get_variants
from app.metrics import BUILTIN_METRICS, metric_values
from app.models import MetricScore, RequestLog
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


@dataclass(frozen=True)
class SeriesPoint:
    n: int
    value: float
    running_mean: float


@dataclass(frozen=True)
class VariantSeries:
    variant_id: int
    label: str
    points: list[SeriesPoint]


def metric_series(session: Session, experiment_id: int, metric: str) -> list[VariantSeries]:
    """Each variant's observations in request order, with the running mean after each one."""
    metric = resolve_metric(metric)
    variants = get_variants(session, experiment_id)
    variant_for_version = {v.version_id: v.id for v in variants}
    if metric in BUILTIN_METRICS:  # plain columns on request_logs; failed requests are skipped
        query = (
            select(RequestLog.id, RequestLog.variant_version_id, getattr(RequestLog, metric))
            .where(RequestLog.experiment_id == experiment_id, RequestLog.error.is_(None))
            .order_by(RequestLog.id)
        )
    else:
        query = (
            select(RequestLog.id, RequestLog.variant_version_id, MetricScore.value)
            .join(MetricScore, MetricScore.request_id == RequestLog.id)
            .where(RequestLog.experiment_id == experiment_id, MetricScore.metric == metric)
            .order_by(RequestLog.id)
        )
    totals = {v.id: 0.0 for v in variants}
    points: dict[int, list[SeriesPoint]] = {v.id: [] for v in variants}
    for _, version_id, value in session.execute(query):
        variant_id = variant_for_version.get(version_id)
        if variant_id is None or value is None:
            continue
        totals[variant_id] += float(value)
        n = len(points[variant_id]) + 1
        points[variant_id].append(SeriesPoint(n, float(value), totals[variant_id] / n))
    return [VariantSeries(v.id, v.label, points[v.id]) for v in variants]