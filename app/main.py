"""Prompt registry HTTP API. No LLM calls in this phase."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
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
from sqlalchemy import select
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


from app.models import ActiveVersion, Experiment, PromptVersion, RequestLog, get_engine

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
        background_tasks.add_task(
            run_judge, make_session, client, log.id, system, body.input, result.text
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