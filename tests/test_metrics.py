"""Tests for metrics: output checks, the async judge, and per-variant summaries."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm import cost_usd
from app.main import app, get_judge_rate, get_llm_client
from app.metrics import JUDGE_MODEL, metric_values, output_metrics, percentile, should_judge
from app.models import MetricScore
from tests.test_completions import MODEL, activate, ask, env, make_prompt  # noqa: F401
from tests.test_experiments import ask_as, create_exp, start

FENCED = '```json\n{"category": "billing", "summary": "Charged twice."}\n```'
BARE = '{"category": "billing", "summary": "Charged twice."}'
GOOD_VERDICT = '{"reasoning": "Correct category.", "score": 4}'


class ScriptedLLM:
    """Fake Anthropic client that answers per model. A list is consumed in order."""

    def __init__(self, by_model):
        self.by_model = by_model
        self.calls = []
        self.messages = self  # client.messages.create(...) lands on self.create

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.by_model[kwargs["model"]]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def script(by_model, rate=1.0):
    """Install a scripted LLM and a judge sampling rate. The env fixture undoes this."""
    llm = ScriptedLLM(by_model)
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_judge_rate] = lambda: rate
    return llm


def connection_error():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def scores(engine, request_id):
    with Session(engine) as session:
        rows = session.scalars(select(MetricScore).where(MetricScore.request_id == request_id))
        return {row.metric: row for row in rows}


def run_experiment(env, n_users=30):
    prompt_id = make_prompt(env.http, "p", ["Version one", "Version two"])
    experiment_id = create_exp(env.http, prompt_id).json()["id"]
    start(env.http, experiment_id)
    for i in range(n_users):
        assert ask_as(env.http, prompt_id, f"user-{i}").status_code == 200
    return experiment_id


def test_output_metrics_distinguish_fenced_bare_and_prose():
    assert output_metrics(BARE) == {"json_valid": 1.0, "json_bare": 1.0}
    assert output_metrics(FENCED) == {"json_valid": 1.0, "json_bare": 0.0}
    assert output_metrics("Sure! Category: billing") == {"json_valid": 0.0, "json_bare": 0.0}


def test_percentile_is_nearest_rank():
    assert percentile(list(range(1, 101)), 95) == 95.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 95) is None


def test_judge_score_is_stored_with_cost_and_reasoning(env):
    llm = script({MODEL: FENCED, JUDGE_MODEL: GOOD_VERDICT})
    prompt_id = make_prompt(env.http, "p", ["Classify the email."])
    assert ask(env.http, prompt_id).status_code == 200

    stored = scores(env.engine, request_id=1)
    assert stored["judge_quality"].value == 4.0
    assert stored["judge_quality"].cost_usd == pytest.approx(cost_usd(100, 20, JUDGE_MODEL))
    assert stored["judge_quality"].details["reasoning"] == "Correct category."
    assert stored["json_valid"].value == 1.0
    assert stored["json_bare"].value == 0.0  # the fence means it is not bare JSON
    assert [call["model"] for call in llm.calls] == [MODEL, JUDGE_MODEL]
    # The judge sees the rendered instructions and the output it must grade.
    judge_prompt = llm.calls[1]["messages"][0]["content"]
    assert "Classify the email." in judge_prompt and "Charged twice." in judge_prompt


def test_unparseable_judge_reply_is_retried_then_recorded_as_error(env):
    llm = script({MODEL: BARE, JUDGE_MODEL: ["not json", "still not json"]})
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    assert ask(env.http, prompt_id).status_code == 200  # the caller never sees judge trouble

    stored = scores(env.engine, request_id=1)
    assert "judge_quality" not in stored
    assert stored["judge_error"].value == 1.0
    assert stored["judge_error"].cost_usd == pytest.approx(2 * cost_usd(100, 20, JUDGE_MODEL))
    assert len(llm.calls) == 3  # one generation plus two judge attempts


def test_judge_retry_can_succeed(env):
    llm = script({MODEL: BARE, JUDGE_MODEL: ["oops", GOOD_VERDICT]})
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    ask(env.http, prompt_id)

    stored = scores(env.engine, request_id=1)
    assert stored["judge_quality"].value == 4.0
    assert stored["judge_quality"].cost_usd == pytest.approx(2 * cost_usd(100, 20, JUDGE_MODEL))
    assert len(llm.calls) == 3


def test_judge_provider_error_does_not_break_the_request(env):
    llm = script({MODEL: BARE, JUDGE_MODEL: [connection_error()]})
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    assert ask(env.http, prompt_id).status_code == 200

    stored = scores(env.engine, request_id=1)
    assert "judge_quality" not in stored
    assert stored["judge_error"].value == 1.0
    assert len(llm.calls) == 2  # provider errors are not retried


def test_sample_rate_zero_skips_the_judge_but_keeps_cheap_metrics(env):
    llm = script({MODEL: FENCED}, rate=0.0)  # a judge call would raise KeyError
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    assert ask(env.http, prompt_id).status_code == 200

    stored = scores(env.engine, request_id=1)
    assert set(stored) == {"json_valid", "json_bare"}
    assert len(llm.calls) == 1


def test_judge_is_skipped_when_it_would_grade_its_own_model(env):
    llm = script({JUDGE_MODEL: FENCED})
    prompt_id = env.http.post(
        "/prompts", json={"name": "p", "actor": "t", "reason": "test"}
    ).json()["id"]
    env.http.post(
        f"/prompts/{prompt_id}/versions",
        json={
            "system_prompt": "Classify.",
            "model": JUDGE_MODEL,
            "temperature": 0.0,
            "commit_message": "t",
            "actor": "t",
            "reason": "test",
        },
    )
    activate(env.http, prompt_id, 1)
    assert ask(env.http, prompt_id).status_code == 200

    assert len(llm.calls) == 1
    assert "judge_quality" not in scores(env.engine, request_id=1)


def test_failed_generation_gets_no_metrics_and_no_judge(env):
    llm = script({MODEL: [connection_error()]})
    prompt_id = make_prompt(env.http, "p", ["Classify."])
    assert ask(env.http, prompt_id).status_code == 502

    assert scores(env.engine, request_id=1) == {}
    assert len(llm.calls) == 1


def test_should_judge_is_deterministic_and_matches_rate():
    picked = [should_judge(i, 0.3) for i in range(1, 5001)]
    assert picked == [should_judge(i, 0.3) for i in range(1, 5001)]
    assert sum(picked) / 5000 == pytest.approx(0.3, abs=0.03)
    assert should_judge(1, 1.0) and not should_judge(1, 0.0)


def test_experiment_metrics_summarise_each_variant(env):
    script({MODEL: FENCED, JUDGE_MODEL: GOOD_VERDICT})
    experiment_id = run_experiment(env)

    body = env.http.get(f"/experiments/{experiment_id}/metrics").json()
    variants = body["variants"]
    assert len(variants) == 2
    assert sum(v["requests"] for v in variants) == 30
    assert sum(v["judge_n"] for v in variants) == 30
    for v in variants:
        assert v["requests"] > 0
        assert v["error_rate"] == 0.0
        assert v["json_valid_rate"] == 1.0
        assert v["json_bare_rate"] == 0.0
        assert v["judge_mean"] == pytest.approx(4.0)
        assert v["judge_std"] == pytest.approx(0.0)
        assert v["judge_failures"] == 0
        assert v["mean_cost_usd"] == pytest.approx(cost_usd(100, 20, MODEL))
        assert v["judge_cost_usd"] == pytest.approx(v["judge_n"] * cost_usd(100, 20, JUDGE_MODEL))
        assert v["p95_latency_ms"] is not None
    assert env.http.get("/experiments/999/metrics").status_code == 404


def test_metric_values_groups_raw_values_by_variant(env):
    script({MODEL: FENCED, JUDGE_MODEL: GOOD_VERDICT})
    experiment_id = run_experiment(env)

    with Session(env.engine) as session:
        quality = metric_values(session, experiment_id, "judge_quality")
        tokens = metric_values(session, experiment_id, "output_tokens")

    assert len(quality) == 2
    assert sum(len(values) for values in quality.values()) == 30
    assert all(x == 4.0 for values in quality.values() for x in values)
    assert all(x == 20.0 for values in tokens.values() for x in values)