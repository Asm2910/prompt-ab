"""Prompt registry HTTP API. No LLM calls in this phase."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.models import get_engine
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
