"""Tests for the list and series endpoints the dashboard uses."""

import pytest

from tests.test_completions import activate, env, make_prompt  # noqa: F401
from tests.test_experiments import create_exp
from tests.test_results import drive, install_judge, new_experiment


def test_prompt_list_shows_version_count_and_active_version(env):
    prompt_id = make_prompt(env.http, "classifier", ["one", "two"])
    empty_id = env.http.post(
        "/prompts", json={"name": "empty", "actor": "t", "reason": "test"}
    ).json()["id"]

    rows = {row["id"]: row for row in env.http.get("/prompts").json()}
    assert rows[prompt_id]["name"] == "classifier"
    assert rows[prompt_id]["versions"] == 2 and rows[prompt_id]["active_version"] == 2
    assert rows[empty_id]["versions"] == 0 and rows[empty_id]["active_version"] is None

    activate(env.http, prompt_id, 1)  # roll back
    rows = {row["id"]: row for row in env.http.get("/prompts").json()}
    assert rows[prompt_id]["active_version"] == 1


def test_experiment_list_is_newest_first_and_includes_variants(env):
    prompt_id = make_prompt(env.http, "p", ["one", "two"])
    first = create_exp(env.http, prompt_id).json()["id"]
    second = create_exp(env.http, prompt_id).json()["id"]

    rows = env.http.get("/experiments").json()
    assert [row["id"] for row in rows] == [second, first]
    assert len(rows[0]["variants"]) == 2


def test_series_is_in_request_order_with_running_means(env):
    install_judge()
    prompt_id, experiment_id = new_experiment(env, target=50)
    drive(env, prompt_id, 40)

    body = env.http.get(f"/experiments/{experiment_id}/series?metric=judge_quality").json()
    assert body["metric"] == "judge_quality"
    assert sum(len(v["points"]) for v in body["variants"]) == 40
    for variant in body["variants"]:
        points = variant["points"]
        assert [p["n"] for p in points] == list(range(1, len(points) + 1))
        mean = sum(p["value"] for p in points) / len(points)
        assert points[-1]["running_mean"] == pytest.approx(mean)
    by_label = {v["label"]: v["points"] for v in body["variants"]}
    assert by_label["v2"][-1]["running_mean"] > by_label["v1"][-1]["running_mean"] + 1


def test_series_works_for_built_in_metrics(env):
    install_judge()
    prompt_id, experiment_id = new_experiment(env, target=50)
    drive(env, prompt_id, 10)

    body = env.http.get(f"/experiments/{experiment_id}/series?metric=output_tokens").json()
    assert sum(len(v["points"]) for v in body["variants"]) == 10
    assert all(
        p["running_mean"] == pytest.approx(20.0) for v in body["variants"] for p in v["points"]
    )


def test_series_rejects_unknown_metric_and_unknown_experiment(env):
    install_judge()
    _, experiment_id = new_experiment(env, target=50)

    assert env.http.get(f"/experiments/{experiment_id}/series?metric=nonsense").status_code == 400
    assert env.http.get("/experiments/999/series").status_code == 404