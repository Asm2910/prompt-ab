"""Tests for POST /v1/completions. A fake LLM replaces the real API."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.main import app, get_db, get_judge_rate, get_llm_client, get_session_factory
from app.models import Base, RequestLog

MODEL = "claude-haiku-4-5-20251001"


class FakeLLM:
    """Stands in for the Anthropic client. Records every call it receives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None
        self.text = "billing"
        self.messages = self  # client.messages.create(...) lands on self.create

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


@pytest.fixture
def env():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    fake = FakeLLM()

    def override_db():
        session = Session(engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_llm_client] = lambda: fake
    app.dependency_overrides[get_judge_rate] = lambda: 0.0  # judging is off unless a test turns it on
    app.dependency_overrides[get_session_factory] = lambda: (lambda: Session(engine))
    yield SimpleNamespace(http=TestClient(app), engine=engine, llm=fake)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved)


def make_prompt(http, name: str, system_prompts: list[str]) -> int:
    """Create a prompt with one version per text and activate the last one."""
    prompt_id = http.post(
        "/prompts", json={"name": name, "actor": "t", "reason": "test"}
    ).json()["id"]
    for text in system_prompts:
        http.post(
            f"/prompts/{prompt_id}/versions",
            json={
                "system_prompt": text,
                "model": MODEL,
                "temperature": 0.0,
                "commit_message": "test",
                "actor": "t",
                "reason": "test",
            },
        )
    activate(http, prompt_id, len(system_prompts))
    return prompt_id


def activate(http, prompt_id: int, version: int) -> None:
    http.post(
        f"/prompts/{prompt_id}/activate",
        json={"version": version, "actor": "t", "reason": "test"},
    )


def logs(engine) -> list[RequestLog]:
    with Session(engine) as session:
        return list(session.scalars(select(RequestLog).order_by(RequestLog.id)))


def ask(http, prompt_id: int, variables: dict | None = None):
    return http.post(
        "/v1/completions",
        json={
            "prompt_id": prompt_id,
            "user_id": "u1",
            "variables": variables or {},
            "input": "I was charged twice",
        },
    )


def test_success_logs_tokens_and_cost(env):
    prompt_id = make_prompt(env.http, "p", ["Classify. Tone: {{tone}}"])
    response = ask(env.http, prompt_id, {"tone": "formal"})

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "billing"
    assert body["version"] == 1
    assert body["cost_usd"] == pytest.approx(0.0002)  # 100 in at $1 + 20 out at $5 per M
    # The variable was filled before the call, and the input went as the user message.
    assert env.llm.calls[0]["system"] == "Classify. Tone: formal"
    assert env.llm.calls[0]["messages"][0]["content"] == "I was charged twice"

    rows = logs(env.engine)
    assert len(rows) == 1
    assert (rows[0].input_tokens, rows[0].output_tokens) == (100, 20)
    assert rows[0].error is None


def test_missing_variable_returns_400_and_makes_no_llm_call(env):
    prompt_id = make_prompt(env.http, "p", ["Tone: {{tone}}"])
    response = ask(env.http, prompt_id, {})

    assert response.status_code == 400
    assert "tone" in response.json()["detail"]
    assert env.llm.calls == []


def test_no_active_version_returns_404(env):
    prompt_id = env.http.post(
        "/prompts", json={"name": "empty", "actor": "t", "reason": "test"}
    ).json()["id"]
    response = ask(env.http, prompt_id)

    assert response.status_code == 404
    assert env.llm.calls == []


def test_llm_error_returns_502_and_is_logged(env):
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    env.llm.error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )
    response = ask(env.http, prompt_id)

    assert response.status_code == 502
    rows = logs(env.engine)
    assert len(rows) == 1  # the failed call still left a log row
    assert rows[0].error is not None
    assert rows[0].output_text is None


def test_rollback_changes_which_version_serves(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])

    ask(env.http, prompt_id)  # version 2 is active
    activate(env.http, prompt_id, 1)  # roll back
    ask(env.http, prompt_id)

    assert env.llm.calls[0]["system"] == "Version two"
    assert env.llm.calls[1]["system"] == "Version one"
    rows = logs(env.engine)
    assert rows[0].version_id != rows[1].version_id



def test_placeholder_with_spaces_is_filled(env):
    prompt_id = make_prompt(env.http, "p", ["Tone: {{ tone }}"])
    response = ask(env.http, prompt_id, {"tone": "formal"})

    assert response.status_code == 200
    assert env.llm.calls[0]["system"] == "Tone: formal"