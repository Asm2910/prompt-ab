"""Tests for experiments and routing through /v1/completions."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog
from tests.test_completions import (  # noqa: F401  (env is a pytest fixture)
    ask,
    env,
    logs,
    make_prompt,
)


def create_exp(http, prompt_id, shares=(0.5, 0.5), versions=(1, 2)):
    return http.post(
        "/experiments",
        json={
            "name": "test-exp",
            "prompt_id": prompt_id,
            "variants": [
                {"version": v, "traffic_share": s} for v, s in zip(versions, shares)
            ],
            "target_sample_size": 100,
            "actor": "t",
            "reason": "test",
        },
    )


def start(http, experiment_id):
    return http.post(
        f"/experiments/{experiment_id}/start", json={"actor": "t", "reason": "test"}
    )


def stop(http, experiment_id, cancel=False):
    return http.post(
        f"/experiments/{experiment_id}/stop",
        json={"actor": "t", "reason": "test", "cancel": cancel},
    )


def ask_as(http, prompt_id, user_id):
    return http.post(
        "/v1/completions",
        json={
            "prompt_id": prompt_id,
            "user_id": user_id,
            "variables": {},
            "input": "I was charged twice",
        },
    )


def test_shares_must_sum_to_one(env):
    prompt_id = make_prompt(env.http, "p", ["one", "two"])
    assert create_exp(env.http, prompt_id, shares=(0.5, 0.4)).status_code == 400


def test_needs_at_least_two_variants(env):
    prompt_id = make_prompt(env.http, "p", ["one", "two"])
    response = create_exp(env.http, prompt_id, shares=(1.0,), versions=(1,))
    assert response.status_code == 400


def test_unknown_version_returns_404(env):
    prompt_id = make_prompt(env.http, "p", ["one", "two"])
    assert create_exp(env.http, prompt_id, versions=(1, 9)).status_code == 404


def test_running_experiment_splits_users_and_logs_variant(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    experiment_id = create_exp(env.http, prompt_id).json()["id"]
    assert start(env.http, experiment_id).status_code == 200

    for i in range(60):
        assert ask_as(env.http, prompt_id, f"user-{i}").status_code == 200

    assert {call["system"] for call in env.llm.calls} == {"Version one", "Version two"}
    rows = logs(env.engine)
    assert len(rows) == 60
    assert all(row.experiment_id == experiment_id for row in rows)
    assert all(row.variant_version_id == row.version_id for row in rows)


def test_same_user_keeps_the_same_variant(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    start(env.http, create_exp(env.http, prompt_id).json()["id"])

    for _ in range(6):
        ask_as(env.http, prompt_id, "user-7")

    assert len({call["system"] for call in env.llm.calls}) == 1


def test_each_user_is_recorded_once(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    experiment_id = create_exp(env.http, prompt_id).json()["id"]
    start(env.http, experiment_id)

    for _ in range(3):
        ask_as(env.http, prompt_id, "user-1")
    ask_as(env.http, prompt_id, "user-2")

    variants = env.http.get(f"/experiments/{experiment_id}").json()["variants"]
    assert sum(v["assigned_users"] for v in variants) == 2


def test_draft_experiment_does_not_change_traffic(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    create_exp(env.http, prompt_id)  # created but never started

    ask(env.http, prompt_id)

    assert env.llm.calls[0]["system"] == "Version two"  # the active version
    assert logs(env.engine)[0].experiment_id is None


def test_only_one_running_experiment_per_prompt(env):
    prompt_id = make_prompt(env.http, "p", ["one", "two"])
    first = create_exp(env.http, prompt_id).json()["id"]
    second = create_exp(env.http, prompt_id).json()["id"]

    assert start(env.http, first).status_code == 200
    assert start(env.http, second).status_code == 409


def test_stopping_returns_traffic_to_the_active_version(env):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    experiment_id = create_exp(env.http, prompt_id).json()["id"]
    start(env.http, experiment_id)
    assert stop(env.http, experiment_id).json()["status"] == "completed"

    ask_as(env.http, prompt_id, "user-1")

    assert env.llm.calls[0]["system"] == "Version two"  # active version again
    assert logs(env.engine)[0].experiment_id is None
    assert stop(env.http, experiment_id).status_code == 409  # already stopped
    with Session(env.engine) as session:
        actions = [row.action for row in session.scalars(select(AuditLog))]
    assert "start_experiment" in actions and "stop_experiment" in actions

