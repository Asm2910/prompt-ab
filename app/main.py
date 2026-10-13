"""Prompt registry HTTP API. No LLM calls in this phase."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from app.results import (
    CUSTOM_METRICS,
    UnknownMetricError,
    analyze_experiment,
    metric_series,
    resolve_metric,
)
from app.stats import required_sample_size

from app.registry import (
    DuplicatePromptError,
    MissingVariablesError,
    PromptNotFoundError,
    VersionNotFoundError,
    activate_version,
    add_version,
    create_prompt,
    get_version,
    list_audit_log,
    list_versions,
    render_template,
)

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.llm import LLMError, complete, cost_usd, get_client  

from app.experiments import (
    ExperimentNotFoundError,
    ExperimentStateError,
    InvalidExperimentError,
    VariantSpec,
    assignment_counts,
    create_experiment,
    get_experiment,
    get_or_assign,
    get_running_experiment,
    get_variants,
    start_experiment,
    stop_experiment,
)

from app.metrics import (
    JUDGE_MODEL,
    add_output_metrics,
    judge_sample_rate,
    run_judge,
    should_judge,
    summarize_experiment,
)

from app.models import (
    ActiveVersion,
    AuditLog,
    Experiment,
    JudgeRubric,
    MetricScore,
    Prompt,
    PromptVersion,
    RequestLog,
    get_engine,
)

from app.registry import (
    DuplicatePromptError,
    MissingVariablesError,
    PromptNotFoundError,
    VersionNotFoundError,
    activate_version,
    add_version,
    create_prompt,
    get_version,
    list_audit_log,
    list_versions,
)

app = FastAPI(title="Prompt A/B")


class CreatePromptBody(BaseModel):
    name: str
    actor: str
    reason: str


class AddVersionBody(BaseModel):
    system_prompt: str
    model: str
    temperature: float
    commit_message: str
    actor: str
    reason: str


class ActivateBody(BaseModel):
    version: int
    actor: str
    reason: str


class PromptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prompt_id: int
    version: int
    system_prompt: str
    model: str
    temperature: float
    commit_message: str
    created_at: datetime


class ActiveOut(BaseModel):
    prompt_id: int
    version: int
    version_id: int


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    actor: str
    action: str
    prompt_id: int
    details: dict[str, Any]
    created_at: datetime


def get_db() -> Generator[Session, None, None]:
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _http_for(exc: Exception) -> HTTPException:
    if isinstance(exc, PromptNotFoundError | VersionNotFoundError | ExperimentNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    if isinstance(exc, DuplicatePromptError | ExperimentStateError):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc))
    if isinstance(exc, MissingVariablesError | InvalidExperimentError):
        return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    raise exc


@app.post("/prompts", response_model=PromptOut)
def post_prompt(body: CreatePromptBody, session: Session = Depends(get_db)) -> PromptOut:
    try:
        prompt = create_prompt(session, body.name, body.actor, body.reason)
    except Exception as exc:
        raise _http_for(exc) from exc
    return PromptOut.model_validate(prompt)


@app.post("/prompts/{prompt_id}/versions", response_model=VersionOut)
def post_version(
    prompt_id: int, body: AddVersionBody, session: Session = Depends(get_db)
) -> VersionOut:
    try:
        row = add_version(
            session,
            prompt_id,
            body.system_prompt,
            body.model,
            body.temperature,
            body.commit_message,
            body.actor,
            body.reason,
        )
    except Exception as exc:
        raise _http_for(exc) from exc
    return VersionOut.model_validate(row)


@app.get("/prompts/{prompt_id}/versions", response_model=list[VersionOut])
def get_versions(prompt_id: int, session: Session = Depends(get_db)) -> list[VersionOut]:
    try:
        rows = list_versions(session, prompt_id)
    except Exception as exc:
        raise _http_for(exc) from exc
    return [VersionOut.model_validate(row) for row in rows]


@app.get("/prompts/{prompt_id}/versions/{version}", response_model=VersionOut)
def get_one_version(
    prompt_id: int, version: int, session: Session = Depends(get_db)
) -> VersionOut:
    try:
        row = get_version(session, prompt_id, version)
    except Exception as exc:
        raise _http_for(exc) from exc
    return VersionOut.model_validate(row)


@app.post("/prompts/{prompt_id}/activate", response_model=ActiveOut)
def post_activate(
    prompt_id: int, body: ActivateBody, session: Session = Depends(get_db)
) -> ActiveOut:
    try:
        active = activate_version(
            session, prompt_id, body.version, body.actor, body.reason
        )
    except Exception as exc:
        raise _http_for(exc) from exc
    return ActiveOut(
        prompt_id=active.prompt_id,
        version=body.version,
        version_id=active.version_id,
    )


@app.get("/audit-log", response_model=list[AuditOut])
def get_audit_log(session: Session = Depends(get_db)) -> list[AuditOut]:
    return [AuditOut.model_validate(row) for row in list_audit_log(session)]


# ---------------------------------------------------------------------------
# Phase 2: serving endpoint
# ---------------------------------------------------------------------------



class CompletionBody(BaseModel):
    prompt_id: int
    user_id: str
    variables: dict[str, str] = Field(default_factory=dict)
    input: str


class CompletionOut(BaseModel):
    request_id: int
    output: str
    version: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: int


def get_llm_client() -> Anthropic:
    # A dependency so tests can swap in a fake client and never call the real API.
    try:
        return get_client()
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

def get_session_factory() -> Callable[[], Session]:
    # Background jobs outlive the request, so they open their own database session.
    return lambda: Session(get_engine())


def get_judge_rate() -> float:
    # A dependency so tests can switch judging on or off.
    return judge_sample_rate()


@app.post("/v1/completions", response_model=CompletionOut)
def post_completion(
    body: CompletionBody,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
    client: Anthropic = Depends(get_llm_client),
    make_session: Callable[[], Session] = Depends(get_session_factory),
    judge_rate: float = Depends(get_judge_rate),
) -> CompletionOut:
    # 1. Pick the version: a running experiment decides it, otherwise the active version.
    experiment = get_running_experiment(session, body.prompt_id)
    if experiment is not None:
        variant = get_or_assign(session, experiment, body.user_id)
        version = session.get(PromptVersion, variant.version_id)
    else:
        version = session.execute(
            select(PromptVersion)
            .join(ActiveVersion, ActiveVersion.version_id == PromptVersion.id)
            .where(ActiveVersion.prompt_id == body.prompt_id)
        ).scalar_one_or_none()
        if version is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No active version for prompt {body.prompt_id}"
            )

    # 2. Fill the versioned template. The template is versioned, not the filled text.
    try:
        system = render_template(version.system_prompt, body.variables)
    except MissingVariablesError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    log = RequestLog(
        prompt_id=version.prompt_id,
        version_id=version.id,
        user_id=body.user_id,
        variables=body.variables,
        input_text=body.input,
        experiment_id=experiment.id if experiment is not None else None,
        variant_version_id=version.id if experiment is not None else None,
    )

    # 3. Call the LLM. On failure we still keep a log row, so we commit before raising:
    #    get_db rolls back on any exception, which would otherwise erase the row.
    try:
        result = complete(client, version.model, system, body.input)
    except LLMError as exc:
        log.error = str(exc)[:1000]
        session.add(log)
        session.commit()
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"LLM call failed: {exc}"
        ) from exc

    # 4. Log tokens, cost and latency, plus the free output checks, then commit.
    log.output_text = result.text
    log.input_tokens = result.input_tokens
    log.output_tokens = result.output_tokens
    log.cost_usd = cost_usd(result.input_tokens, result.output_tokens, version.model)
    log.latency_ms = result.latency_ms
    session.add(log)
    session.flush()  # assigns log.id, which the metric rows point at
    add_output_metrics(session, log.id, result.text)
    session.commit()

    # 5. The judge runs after the response is sent, so the caller never waits for it.
    #    It is skipped when it would grade its own model, and sampled by JUDGE_SAMPLE_RATE.
    if version.model != JUDGE_MODEL and should_judge(log.id, judge_rate):
        # A rubric fixed per prompt means every variant is graded against the same standard.
        # Without one, the judge grades each output against its own variant's instructions.
        rubric = session.get(JudgeRubric, version.prompt_id)
        judge_task = rubric.rubric if rubric is not None else system
        background_tasks.add_task(
            run_judge, make_session, client, log.id, judge_task, body.input, result.text
        )

    return CompletionOut(
        request_id=log.id,
        output=result.text,
        version=version.version,
        model=version.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=log.cost_usd,
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# Phase 3: experiments
# ---------------------------------------------------------------------------


class VariantIn(BaseModel):
    version: int
    traffic_share: float
    label: str | None = None


class CreateExperimentBody(BaseModel):
    name: str
    prompt_id: int
    variants: list[VariantIn]
    primary_metric: str = "quality"
    target_sample_size: int
    actor: str
    reason: str


class ActorBody(BaseModel):
    actor: str
    reason: str


class StopBody(ActorBody):
    cancel: bool = False  # true marks the experiment cancelled instead of completed


class VariantOut(BaseModel):
    id: int
    version_id: int
    label: str
    traffic_share: float
    assigned_users: int


class ExperimentOut(BaseModel):
    id: int
    name: str
    prompt_id: int
    status: str
    primary_metric: str
    target_sample_size: int
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    variants: list[VariantOut]


def _experiment_out(session: Session, experiment: Experiment) -> ExperimentOut:
    counts = assignment_counts(session, experiment.id)
    variants = [
        VariantOut(
            id=v.id,
            version_id=v.version_id,
            label=v.label,
            traffic_share=v.traffic_share,
            assigned_users=counts.get(v.id, 0),
        )
        for v in get_variants(session, experiment.id)
    ]
    return ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        prompt_id=experiment.prompt_id,
        status=experiment.status,
        primary_metric=experiment.primary_metric,
        target_sample_size=experiment.target_sample_size,
        created_at=experiment.created_at,
        started_at=experiment.started_at,
        ended_at=experiment.ended_at,
        variants=variants,
    )


@app.post("/experiments", response_model=ExperimentOut)
def post_experiment(
    body: CreateExperimentBody, session: Session = Depends(get_db)
) -> ExperimentOut:
    specs = [VariantSpec(v.version, v.traffic_share, v.label) for v in body.variants]
    try:
        experiment = create_experiment(
            session, body.name, body.prompt_id, specs, body.primary_metric,
            body.target_sample_size, body.actor, body.reason,
        )
    except Exception as exc:
        raise _http_for(exc) from exc
    return _experiment_out(session, experiment)


@app.get("/experiments/{experiment_id}", response_model=ExperimentOut)
def get_one_experiment(experiment_id: int, session: Session = Depends(get_db)) -> ExperimentOut:
    try:
        experiment = get_experiment(session, experiment_id)
    except Exception as exc:
        raise _http_for(exc) from exc
    return _experiment_out(session, experiment)


@app.post("/experiments/{experiment_id}/start", response_model=ExperimentOut)
def post_start_experiment(
    experiment_id: int, body: ActorBody, session: Session = Depends(get_db)
) -> ExperimentOut:
    try:
        experiment = start_experiment(session, experiment_id, body.actor, body.reason)
    except Exception as exc:
        raise _http_for(exc) from exc
    return _experiment_out(session, experiment)


@app.post("/experiments/{experiment_id}/stop", response_model=ExperimentOut)
def post_stop_experiment(
    experiment_id: int, body: StopBody, session: Session = Depends(get_db)
) -> ExperimentOut:
    try:
        experiment = stop_experiment(
            session, experiment_id, body.actor, body.reason, cancel=body.cancel
        )
    except Exception as exc:
        raise _http_for(exc) from exc
    return _experiment_out(session, experiment)

    # ---------------------------------------------------------------------------
# Phase 4: metrics
# ---------------------------------------------------------------------------


class VariantMetricsOut(BaseModel):
    variant_id: int
    label: str
    version_id: int
    requests: int
    errors: int
    error_rate: float | None
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_cost_usd: float | None
    total_cost_usd: float
    json_valid_rate: float | None
    json_bare_rate: float | None
    judge_n: int
    judge_mean: float | None
    judge_std: float | None
    judge_failures: int
    judge_cost_usd: float


class ExperimentMetricsOut(BaseModel):
    experiment_id: int
    status: str
    target_sample_size: int
    variants: list[VariantMetricsOut]


@app.get("/experiments/{experiment_id}/metrics", response_model=ExperimentMetricsOut)
def get_experiment_metrics(
    experiment_id: int, session: Session = Depends(get_db)
) -> ExperimentMetricsOut:
    try:
        experiment = get_experiment(session, experiment_id)
    except Exception as exc:
        raise _http_for(exc) from exc
    summaries = summarize_experiment(session, experiment_id)
    return ExperimentMetricsOut(
        experiment_id=experiment.id,
        status=experiment.status,
        target_sample_size=experiment.target_sample_size,
        variants=[VariantMetricsOut(**asdict(s)) for s in summaries],
    )

    # ---------------------------------------------------------------------------
# Phase 5: statistics
# ---------------------------------------------------------------------------


class VariantStatsOut(BaseModel):
    variant_id: int
    label: str
    n: int
    mean: float | None
    std: float | None
    samples_needed: int


class ComparisonOut(BaseModel):
    variant_id: int
    label: str
    verdict: str  # pending | better | worse | no_difference | insufficient_data
    diff: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float | None
    p_welch: float | None
    p_mann_whitney: float | None
    p_decision: float | None
    p_adjusted: float | None
    cohens_d: float | None
    mde: float | None


class ExperimentResultsOut(BaseModel):
    experiment_id: int
    experiment_status: str
    metric: str
    higher_is_better: bool
    decision_test: str
    alpha: float
    power: float
    target_per_variant: int
    decision: str  # not_ready | winner | control_holds | no_difference
    winner: str | None
    interim: bool
    variants: list[VariantStatsOut]
    comparisons: list[ComparisonOut]
    note: str


class SampleSizeOut(BaseModel):
    per_variant: int
    std: float
    effect: float
    alpha: float
    power: float


@app.get("/experiments/{experiment_id}/results", response_model=ExperimentResultsOut)
def get_experiment_results(
    experiment_id: int,
    alpha: float = Query(0.05, gt=0, lt=0.5),
    metric: str | None = None,
    session: Session = Depends(get_db),
) -> ExperimentResultsOut:
    try:
        results = analyze_experiment(session, experiment_id, alpha=alpha, metric=metric)
    except UnknownMetricError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        raise _http_for(exc) from exc
    return ExperimentResultsOut(**asdict(results))


@app.get("/stats/sample-size", response_model=SampleSizeOut)
def get_sample_size(
    std: float = Query(..., gt=0),
    effect: float = Query(..., gt=0),
    alpha: float = Query(0.05, gt=0, lt=0.5),
    power: float = Query(0.8, gt=0, lt=1),
) -> SampleSizeOut:
    """How many observations each variant needs. Decide this BEFORE starting an experiment."""
    return SampleSizeOut(
        per_variant=required_sample_size(std, effect, alpha, power),
        std=std,
        effect=effect,
        alpha=alpha,
        power=power,
    )


# ---------------------------------------------------------------------------
# Phase 6: endpoints the dashboard needs
# ---------------------------------------------------------------------------


class PromptSummaryOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    versions: int
    active_version: int | None


class SeriesPointOut(BaseModel):
    n: int
    value: float
    running_mean: float


class VariantSeriesOut(BaseModel):
    variant_id: int
    label: str
    points: list[SeriesPointOut]


class SeriesOut(BaseModel):
    experiment_id: int
    metric: str
    variants: list[VariantSeriesOut]


@app.get("/prompts", response_model=list[PromptSummaryOut])
def list_prompts(session: Session = Depends(get_db)) -> list[PromptSummaryOut]:
    out = []
    for prompt in session.scalars(select(Prompt).order_by(Prompt.id)):
        count = session.scalar(
            select(func.count()).select_from(PromptVersion).where(PromptVersion.prompt_id == prompt.id)
        )
        active = session.execute(
            select(PromptVersion.version)
            .join(ActiveVersion, ActiveVersion.version_id == PromptVersion.id)
            .where(ActiveVersion.prompt_id == prompt.id)
        ).scalar_one_or_none()
        out.append(
            PromptSummaryOut(
                id=prompt.id,
                name=prompt.name,
                created_at=prompt.created_at,
                versions=count or 0,
                active_version=active,
            )
        )
    return out


@app.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(session: Session = Depends(get_db)) -> list[ExperimentOut]:
    rows = session.scalars(select(Experiment).order_by(Experiment.id.desc())).all()
    return [_experiment_out(session, row) for row in rows]


@app.get("/experiments/{experiment_id}/series", response_model=SeriesOut)
def get_experiment_series(
    experiment_id: int, metric: str | None = None, session: Session = Depends(get_db)
) -> SeriesOut:
    try:
        experiment = get_experiment(session, experiment_id)
        name = resolve_metric(metric or experiment.primary_metric)
        series = metric_series(session, experiment_id, name)
    except UnknownMetricError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        raise _http_for(exc) from exc
    return SeriesOut(
        experiment_id=experiment_id,
        metric=name,
        variants=[VariantSeriesOut(**asdict(s)) for s in series],
    )

# ---------------------------------------------------------------------------
# Phase 7: fixed judge rubric and client-recorded metrics
# ---------------------------------------------------------------------------


class RubricBody(BaseModel):
    rubric: str = Field(min_length=20)
    actor: str
    reason: str


class RubricOut(BaseModel):
    prompt_id: int
    rubric: str | None


class MetricIn(BaseModel):
    metric: str
    value: float
    note: str | None = None


class MetricPostedOut(BaseModel):
    request_id: int
    metric: str
    value: float


@app.put("/prompts/{prompt_id}/rubric", response_model=RubricOut)
def put_rubric(
    prompt_id: int, body: RubricBody, session: Session = Depends(get_db)
) -> RubricOut:
    if session.get(Prompt, prompt_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"prompt {prompt_id} not found")
    row = session.get(JudgeRubric, prompt_id)
    if row is None:
        row = JudgeRubric(prompt_id=prompt_id, rubric=body.rubric)
        session.add(row)
    else:
        row.rubric = body.rubric
    session.add(
        AuditLog(
            actor=body.actor,
            action="set_rubric",
            prompt_id=prompt_id,
            details={"reason": body.reason, "characters": len(body.rubric)},
        )
    )
    session.flush()
    return RubricOut(prompt_id=prompt_id, rubric=row.rubric)


@app.get("/prompts/{prompt_id}/rubric", response_model=RubricOut)
def get_rubric(prompt_id: int, session: Session = Depends(get_db)) -> RubricOut:
    if session.get(Prompt, prompt_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"prompt {prompt_id} not found")
    row = session.get(JudgeRubric, prompt_id)
    return RubricOut(prompt_id=prompt_id, rubric=row.rubric if row else None)


@app.post(
    "/requests/{request_id}/metrics",
    response_model=MetricPostedOut,
    status_code=status.HTTP_201_CREATED,
)
def post_request_metric(
    request_id: int, body: MetricIn, session: Session = Depends(get_db)
) -> MetricPostedOut:
    """Record a metric the caller measured, such as accuracy against a known label."""
    if body.metric not in CUSTOM_METRICS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"metric must be one of: {', '.join(sorted(CUSTOM_METRICS))}",
        )
    low, high = CUSTOM_METRICS[body.metric]
    if not low <= body.value <= high:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{body.metric} must be between {low:g} and {high:g}"
        )
    if session.get(RequestLog, request_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"request {request_id} not found")
    already = session.scalar(
        select(MetricScore.id).where(
            MetricScore.request_id == request_id, MetricScore.metric == body.metric
        )
    )
    if already is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"{body.metric} is already recorded for request {request_id}"
        )
    session.add(
        MetricScore(
            request_id=request_id,
            metric=body.metric,
            value=body.value,
            details={"note": body.note} if body.note else None,
        )
    )
    return MetricPostedOut(request_id=request_id, metric=body.metric, value=body.value)