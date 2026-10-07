"""Experiments: define variants, start and stop them, assign users to variants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditLog, Experiment, ExperimentAssignment, ExperimentVariant
from app.registry import get_version
from app.splitter import assign_variant


class ExperimentNotFoundError(LookupError):
    pass


class InvalidExperimentError(ValueError):
    """The experiment definition is wrong (bad shares, too few variants...)."""


class ExperimentStateError(ValueError):
    """The action is not allowed in the experiment's current status."""


@dataclass(frozen=True)
class VariantSpec:
    version: int  # prompt version number, not the database id
    traffic_share: float
    label: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _audit(
    session: Session, actor: str, action: str, prompt_id: int, reason: str, **extra: Any
) -> None:
    session.add(
        AuditLog(
            actor=actor,
            action=action,
            prompt_id=prompt_id,
            details={"reason": reason, **extra},
        )
    )


def create_experiment(
    session: Session,
    name: str,
    prompt_id: int,
    variants: list[VariantSpec],
    primary_metric: str,
    target_sample_size: int,
    actor: str,
    reason: str,
) -> Experiment:
    if len(variants) < 2:
        raise InvalidExperimentError("an experiment needs at least two variants")
    if any(v.traffic_share <= 0 for v in variants):
        raise InvalidExperimentError("every traffic share must be greater than 0")
    if abs(sum(v.traffic_share for v in variants) - 1.0) > 1e-6:
        raise InvalidExperimentError("traffic shares must sum to 1.0")
    if len({v.version for v in variants}) != len(variants):
        raise InvalidExperimentError("each variant must use a different prompt version")
    if target_sample_size < 1:
        raise InvalidExperimentError("target_sample_size must be at least 1")

    # get_version also checks that the prompt exists (raises not-found errors).
    rows = [get_version(session, prompt_id, v.version) for v in variants]

    experiment = Experiment(
        name=name,
        prompt_id=prompt_id,
        status="draft",
        primary_metric=primary_metric,
        target_sample_size=target_sample_size,
    )
    session.add(experiment)
    session.flush()
    for spec, row in zip(variants, rows):
        session.add(
            ExperimentVariant(
                experiment_id=experiment.id,
                version_id=row.id,
                label=spec.label or f"v{spec.version}",
                traffic_share=spec.traffic_share,
            )
        )
    _audit(session, actor, "create_experiment", prompt_id, reason, experiment_id=experiment.id)
    session.flush()
    return experiment


def get_experiment(session: Session, experiment_id: int) -> Experiment:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise ExperimentNotFoundError(f"experiment {experiment_id} not found")
    return experiment


def get_variants(session: Session, experiment_id: int) -> list[ExperimentVariant]:
    # Ordered by id so the hash walk always sees the variants in the same order.
    return list(
        session.scalars(
            select(ExperimentVariant)
            .where(ExperimentVariant.experiment_id == experiment_id)
            .order_by(ExperimentVariant.id)
        ).all()
    )


def get_running_experiment(session: Session, prompt_id: int) -> Experiment | None:
    return session.scalar(
        select(Experiment).where(
            Experiment.prompt_id == prompt_id, Experiment.status == "running"
        )
    )


def start_experiment(session: Session, experiment_id: int, actor: str, reason: str) -> Experiment:
    experiment = get_experiment(session, experiment_id)
    if experiment.status != "draft":
        raise ExperimentStateError(
            f"experiment {experiment_id} is {experiment.status}; only a draft can be started"
        )
    if get_running_experiment(session, experiment.prompt_id) is not None:
        raise ExperimentStateError(
            f"prompt {experiment.prompt_id} already has a running experiment"
        )
    experiment.status = "running"
    experiment.started_at = _utcnow()
    _audit(session, actor, "start_experiment", experiment.prompt_id, reason, experiment_id=experiment.id)
    session.flush()
    return experiment


def stop_experiment(
    session: Session, experiment_id: int, actor: str, reason: str, cancel: bool = False
) -> Experiment:
    experiment = get_experiment(session, experiment_id)
    if experiment.status != "running":
        raise ExperimentStateError(
            f"experiment {experiment_id} is {experiment.status}; only a running one can be stopped"
        )
    experiment.status = "cancelled" if cancel else "completed"
    experiment.ended_at = _utcnow()
    _audit(session, actor, "stop_experiment", experiment.prompt_id, reason,
           experiment_id=experiment.id, outcome=experiment.status)
    session.flush()
    return experiment


def get_or_assign(session: Session, experiment: Experiment, user_id: str) -> ExperimentVariant:
    """Pick this user's variant by hash, and record the assignment the first time."""
    variants = get_variants(session, experiment.id)
    by_id = {v.id: v for v in variants}
    chosen_id = assign_variant(experiment.id, user_id, [(v.id, v.traffic_share) for v in variants])

    already = session.scalar(
        select(ExperimentAssignment.id).where(
            ExperimentAssignment.experiment_id == experiment.id,
            ExperimentAssignment.user_id == user_id,
        )
    )
    if already is None:
        try:
            session.add(
                ExperimentAssignment(
                    experiment_id=experiment.id, user_id=user_id, variant_id=chosen_id
                )
            )
            session.flush()
        except IntegrityError:
            # A concurrent request logged this user first. The hash gives the same
            # variant anyway, so we only discard our duplicate insert.
            session.rollback()
    return by_id[chosen_id]


def assignment_counts(session: Session, experiment_id: int) -> dict[int, int]:
    rows = session.execute(
        select(ExperimentAssignment.variant_id, func.count())
        .where(ExperimentAssignment.experiment_id == experiment_id)
        .group_by(ExperimentAssignment.variant_id)
    ).all()
    return {variant_id: n for variant_id, n in rows}