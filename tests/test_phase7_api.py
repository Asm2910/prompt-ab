"""Tests for the fixed judge rubric and client-recorded metrics such as accuracy."""

from app.metrics import JUDGE_MODEL
from tests.test_completions import MODEL, ask, env, make_prompt  # noqa: F401
from tests.test_experiments import ask_as
from tests.test_metrics import BARE, GOOD_VERDICT, script
from tests.test_results import new_experiment

RUBRIC = "Task: classify the email. 5 means the category is correct."


def put_rubric(http, prompt_id, text=RUBRIC):
    return http.put(
        f"/prompts/{prompt_id}/rubric", json={"rubric": text, "actor": "t", "reason": "test"}
    )


def post_metric(http, request_id, metric="accuracy", value=1.0):
    return http.post(
        f"/requests/{request_id}/metrics", json={"metric": metric, "value": value, "note": "test"}
    )


def test_rubric_replaces_the_variant_instructions_in_the_judge_prompt(env):
    llm = script({MODEL: BARE, JUDGE_MODEL: GOOD_VERDICT})
    prompt_id = make_prompt(env.http, "p", ["Version one"])
    assert put_rubric(env.http, prompt_id).status_code == 200

    assert ask(env.http, prompt_id).status_code == 200

    judge_prompt = llm.calls[1]["messages"][0]["content"]
    assert RUBRIC in judge_prompt
    assert "Version one" not in judge_prompt  # the variant's own instructions are not shown


def test_rubric_can_be_read_updated_and_is_audited(env):
    prompt_id = make_prompt(env.http, "p", ["one"])
    assert env.http.get(f"/prompts/{prompt_id}/rubric").json()["rubric"] is None

    put_rubric(env.http, prompt_id)
    second = "Task: classify the email. 5 means correct, 1 means wrong."
    put_rubric(env.http, prompt_id, second)

    assert env.http.get(f"/prompts/{prompt_id}/rubric").json()["rubric"] == second
    actions = [row["action"] for row in env.http.get("/audit-log").json()]
    assert actions.count("set_rubric") == 2


def test_rubric_rejects_unknown_prompt_and_tiny_rubric(env):
    assert put_rubric(env.http, 999).status_code == 404
    assert env.http.get("/prompts/999/rubric").status_code == 404
    prompt_id = make_prompt(env.http, "p", ["one"])
    assert put_rubric(env.http, prompt_id, "short").status_code == 422


def test_client_metric_validation(env):
    script({MODEL: BARE}, rate=0.0)
    prompt_id = make_prompt(env.http, "p", ["one"])
    request_id = ask(env.http, prompt_id).json()["request_id"]

    assert post_metric(env.http, request_id).status_code == 201
    assert post_metric(env.http, request_id).status_code == 409  # already recorded
    assert post_metric(env.http, request_id, "mystery").status_code == 400  # not a client metric
    assert post_metric(env.http, request_id, "user_rating", 9).status_code == 400  # out of range
    assert post_metric(env.http, 999).status_code == 404


def test_accuracy_drives_the_verdict_like_any_other_metric(env):
    script({MODEL: BARE}, rate=0.0)
    prompt_id, experiment_id = new_experiment(env, target=20)

    for i in range(60):
        body = ask_as(env.http, prompt_id, f"user-{i}").json()
        # Version 2 is right four times in five, version 1 only one time in five.
        right = (i % 5 != 0) if body["version"] == 2 else (i % 5 == 0)
        assert post_metric(env.http, body["request_id"], value=1.0 if right else 0.0).status_code == 201

    result = env.http.get(f"/experiments/{experiment_id}/results?metric=accuracy").json()
    assert result["metric"] == "accuracy" and result["higher_is_better"] is True
    assert result["decision_test"] == "welch"
    assert result["decision"] == "winner" and result["winner"] == "v2"
    assert result["comparisons"][0]["p_adjusted"] < 0.001