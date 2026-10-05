from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.models import ActiveVersion
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


def _prompt(session: Session) -> int:
    return create_prompt(session, "support-email", "alice", "initial").id


def test_create_prompt(session: Session) -> None:
    prompt = create_prompt(session, "billing", "alice", "new prompt")
    assert prompt.id is not None
    assert prompt.name == "billing"
    session.commit()
    audit = list_audit_log(session)
    assert audit[0].action == "create_prompt"
    assert audit[0].actor == "alice"
    assert audit[0].details["reason"] == "new prompt"
    with pytest.raises(DuplicatePromptError):
        create_prompt(session, "billing", "alice", "again")


def test_add_version_auto_increments(session: Session) -> None:
    prompt_id = _prompt(session)
    first = add_version(
        session, prompt_id, "You are v1.", "claude-haiku-4-5-20251001", 0.0, "v1", "alice", "first"
    )
    second = add_version(
        session, prompt_id, "You are v2.", "claude-haiku-4-5-20251001", 0.2, "v2", "alice", "second"
    )
    assert first.version == 1
    assert second.version == 2


def test_list_versions(session: Session) -> None:
    prompt_id = _prompt(session)
    add_version(session, prompt_id, "a", "m", 0.0, "one", "alice", "a")
    add_version(session, prompt_id, "b", "m", 0.0, "two", "alice", "b")
    versions = list_versions(session, prompt_id)
    assert [row.version for row in versions] == [1, 2]


def test_get_version(session: Session) -> None:
    prompt_id = _prompt(session)
    add_version(session, prompt_id, "sys", "m", 0.1, "msg", "alice", "add")
    row = get_version(session, prompt_id, 1)
    assert row.system_prompt == "sys"
    assert row.temperature == 0.1
    with pytest.raises(VersionNotFoundError):
        get_version(session, prompt_id, 9)
    with pytest.raises(PromptNotFoundError):
        get_version(session, 999, 1)


def test_activate_older_version_rolls_back(session: Session) -> None:
    prompt_id = _prompt(session)
    v1 = add_version(session, prompt_id, "old", "m", 0.0, "v1", "alice", "a")
    v2 = add_version(session, prompt_id, "new", "m", 0.0, "v2", "alice", "b")
    activate_version(session, prompt_id, 2, "alice", "ship v2")
    active = session.get(ActiveVersion, prompt_id)
    assert active is not None
    assert active.version_id == v2.id
    activate_version(session, prompt_id, 1, "alice", "rollback")
    active = session.get(ActiveVersion, prompt_id)
    assert active is not None
    assert active.version_id == v1.id


def test_activate_writes_audit_row(session: Session) -> None:
    prompt_id = _prompt(session)
    add_version(session, prompt_id, "sys", "m", 0.0, "v1", "alice", "add")
    activate_version(session, prompt_id, 1, "bob", "go live")
    session.commit()
    rows = [row for row in list_audit_log(session) if row.action == "activate_version"]
    assert len(rows) == 1
    assert rows[0].actor == "bob"
    assert rows[0].details["reason"] == "go live"
    assert rows[0].details["version"] == 1


def test_render_template_fills_placeholders() -> None:
    text = render_template("Hello {{name}}, re {{topic}}.", {"name": "Ada", "topic": "billing"})
    assert text == "Hello Ada, re billing."


def test_render_template_rejects_missing_variable() -> None:
    with pytest.raises(MissingVariablesError, match="user") as exc_info:
        render_template("Q: {{context}}\nUser: {{user}}", {"context": "outage"})
    assert exc_info.value.missing == ["user"]
