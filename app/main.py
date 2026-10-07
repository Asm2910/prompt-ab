"""Prompt registry HTTP API. No LLM calls in this phase."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
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
from app.models import ActiveVersion, PromptVersion, RequestLog, get_engine

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
    if isinstance(exc, PromptNotFoundError | VersionNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    if isinstance(exc, DuplicatePromptError):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc))
    if isinstance(exc, MissingVariablesError):
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




@app.post("/v1/completions", response_model=CompletionOut)
def post_completion(
    body: CompletionBody,
    session: Session = Depends(get_db),
    client: Anthropic = Depends(get_llm_client),
) -> CompletionOut:
    # 1. Find the active version of this prompt.
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

    # 4. Log tokens, cost and latency, then return.
    log.output_text = result.text
    log.input_tokens = result.input_tokens
    log.output_tokens = result.output_tokens
    log.cost_usd = cost_usd(result.input_tokens, result.output_tokens, version.model)
    log.latency_ms = result.latency_ms
    session.add(log)
    session.commit()

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