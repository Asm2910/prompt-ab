"""Prompt registry: versions, activation (rollback), templates, and audit rows."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ActiveVersion, AuditLog, Prompt, PromptVersion

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptNotFoundError(LookupError):
    pass


class VersionNotFoundError(LookupError):
    pass


class DuplicatePromptError(ValueError):
    pass


class MissingVariablesError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"missing template variables: {', '.join(missing)}")


def _require_prompt(session: Session, prompt_id: int) -> Prompt:
    prompt = session.get(Prompt, prompt_id)
    if prompt is None:
        raise PromptNotFoundError(f"prompt {prompt_id} not found")
    return prompt


def _write_audit(
    session: Session,
    actor: str,
    action: str,
    prompt_id: int,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    details: dict[str, Any] = {"reason": reason}
    if extra:
        details.update(extra)
    session.add(
        AuditLog(actor=actor, action=action, prompt_id=prompt_id, details=details)
    )


def create_prompt(session: Session, name: str, actor: str, reason: str) -> Prompt:
    existing = session.scalar(select(Prompt).where(Prompt.name == name))
    if existing is not None:
        raise DuplicatePromptError(f"prompt name {name!r} already exists")
    prompt = Prompt(name=name)
    session.add(prompt)
    session.flush()
    _write_audit(session, actor, "create_prompt", prompt.id, reason, {"name": name})
    session.flush()
    return prompt


def add_version(
    session: Session,
    prompt_id: int,
    system_prompt: str,
    model: str,
    temperature: float,
    commit_message: str,
    actor: str,
    reason: str,
) -> PromptVersion:
    _require_prompt(session, prompt_id)
    # Next version is max(existing)+1 for this prompt only, so prompts do not share a counter.
    max_version = session.scalar(
        select(func.max(PromptVersion.version)).where(
            PromptVersion.prompt_id == prompt_id
        )
    )
    version = (max_version or 0) + 1
    row = PromptVersion(
        prompt_id=prompt_id,
        version=version,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        commit_message=commit_message,
    )
    session.add(row)
    session.flush()
    _write_audit(
        session,
        actor,
        "add_version",
        prompt_id,
        reason,
        {"version": version, "commit_message": commit_message},
    )
    session.flush()
    return row


def list_versions(session: Session, prompt_id: int) -> list[PromptVersion]:
    _require_prompt(session, prompt_id)
    return list(
        session.scalars(
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version)
        ).all()
    )


def get_version(session: Session, prompt_id: int, version: int) -> PromptVersion:
    _require_prompt(session, prompt_id)
    row = session.scalar(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == version,
        )
    )
    if row is None:
        raise VersionNotFoundError(
            f"prompt {prompt_id} version {version} not found"
        )
    return row


def activate_version(
    session: Session, prompt_id: int, version: int, actor: str, reason: str
) -> ActiveVersion:
    # Rollback is the same path: point the single active row at an older version_id.
    row = get_version(session, prompt_id, version)
    active = session.get(ActiveVersion, prompt_id)
    if active is None:
        active = ActiveVersion(prompt_id=prompt_id, version_id=row.id)
        session.add(active)
    else:
        active.version_id = row.id
    _write_audit(
        session,
        actor,
        "activate_version",
        prompt_id,
        reason,
        {"version": version, "version_id": row.id},
    )
    session.flush()
    return active


def list_audit_log(session: Session) -> list[AuditLog]:
    return list(
        session.scalars(select(AuditLog).order_by(AuditLog.created_at.desc(), AuditLog.id.desc())).all()
    )


def render_template(template: str, variables: dict[str, str]) -> str:
    needed = set(_PLACEHOLDER.findall(template))
    missing = sorted(name for name in needed if name not in variables)
    if missing:
        raise MissingVariablesError(missing)
    return _PLACEHOLDER.sub(lambda match: str(variables[match.group(1)]), template)
