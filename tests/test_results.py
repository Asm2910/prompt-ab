"""API tests for GET /experiments/{id}/results and the sample-size calculator."""

import itertools
import json
from types import SimpleNamespace

import pytest

from app.main import app, get_judge_rate, get_llm_client
from app.metrics import JUDGE_MODEL
from tests.test_completions import env, make_prompt  # noqa: F401
from tests.test_experiments import ask_as, start

FENCED = '```json\n{"category": "billing", "summary": "Charged twice."}\n```'


class VariantAwareLLM:
    """Fake client. The judge scores 'Version two' about 2.2 points higher than 'Version one'."""

    def __init__(self):
        self.messages = self  # client.messages.create(...) lands on self.create
        self.scores = {
            "Version one": itertools.cycle([2, 3, 2, 3, 2]),
            "Version two": itertools.cycle([5, 4, 5, 5, 4]),
        }

    def create(self, **kwargs):
        if kwargs["model"] == JUDGE_MODEL:
            # The judge prompt contains the rendered instructions, so it reveals the variant.
            prompt = kwargs["messages"][0]["content"]
            arm = "Version two" if "Version two" in prompt else "Version one"
            text = json.dumps({"reasoning": "scripted", "score": next(self.scores[arm])})
        else:
            text = FENCED
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def install_judge():
    llm = VariantAwareLLM()
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_judge_rate] = lambda: 1.0
    return llm


def new_experiment(env, target):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    experiment_id = env.http.post(
        "/experiments",
        json={
            "name": "stats-exp",
            "prompt_id": prompt_id,
            "variants": [
                {"version": 1, "traffic_share": 0.5},
                {"version": 2, "traffic_share": 0.5},
            ],
            "target_sample_size": target,
            "actor": "t",
            "reason": "test",
        },
    ).json()["id"]
    start(env.http, experiment_id)
    return prompt_id, experiment_id


def drive(env, prompt_id, users):
    for i in range(users):
        assert ask_as(env.http, prompt_id, f"user-{i}").status_code == 200


def test_no_verdict_before_the_target_sample_size(env):
    install_judge()
    prompt_id, experiment_id = new_experiment(env, target=50)
    drive(env, prompt_id, 30)

    body = env.http.get(f"/experiments/{experiment_id}/results").json()
    assert body["metric"] == "judge_quality"  # "quality" is an alias for it
    assert body["decision"] == "not_ready"
    assert body["interim"] is True and body["winner"] is None
    assert [c["verdict"] for c in body["comparisons"]] == ["pending"]
    assert sum(v["n"] for v in body["variants"]) == 30
    assert all(v["samples_needed"] > 0 for v in body["variants"])
    # The numbers are shown for information, but they are not a decision.
    assert body["comparisons"][0]["p_decision"] is not None


def test_clear_winner_is_declared_at_the_target_sample_size(env):
    install_judge()
    prompt_id, experiment_id = new_experiment(env, target=50)
    drive(env, prompt_id, 160)

    body = env.http.get(f"/experiments/{experiment_id}/results").json()
    assert body["decision"] == "winner" and body["interim"] is False
    assert body["winner"] == "v2"
    comparison = body["comparisons"][0]
    assert comparison["verdict"] == "better"
    assert comparison["p_adjusted"] < 0.001
    assert comparison["ci_low"] > 0
    assert comparison["diff"] == pytest.approx(2.2, abs=0.5)


def test_experiment_with_no_data_does_not_crash(env):
    install_judge()
    _, experiment_id = new_experiment(env, target=50)

    body = env.http.get(f"/experiments/{experiment_id}/results").json()
    assert body["decision"] == "not_ready"
    assert body["comparisons"][0]["verdict"] == "insufficient_data"
    assert all(v["n"] == 0 and v["mean"] is None for v in body["variants"])


def test_unknown_metric_and_unknown_experiment(env):
    install_judge()
    _, experiment_id = new_experiment(env, target=50)

    assert env.http.get(f"/experiments/{experiment_id}/results?metric=nonsense").status_code == 400
    assert env.http.get("/experiments/999/results").status_code == 404


def test_latency_is_lower_is_better_and_uses_the_rank_test(env):
    install_judge()
    prompt_id, experiment_id = new_experiment(env, target=50)
    drive(env, prompt_id, 20)

    body = env.http.get(f"/experiments/{experiment_id}/results?metric=latency_ms").json()
    assert body["metric"] == "latency_ms"
    assert body["higher_is_better"] is False
    assert body["decision_test"] == "mann_whitney"


def test_sample_size_calculator(env):
    body = env.http.get("/stats/sample-size?std=1&effect=0.5").json()
    assert body["per_variant"] == 63
    assert env.http.get("/stats/sample-size?std=1&effect=0").status_code == 422
    assert env.http.get("/stats/sample-size?std=1").status_code == 422