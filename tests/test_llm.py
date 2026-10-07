"""Tests for app/llm.py. A fake client stands in for the real API."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.llm import LLMError, complete, cost_usd


class FakeMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


def fake_client(response=None, error=None):
    return SimpleNamespace(messages=FakeMessages(response, error))


def test_cost_known_model():
    # Haiku: 1M input at $1 plus 1M output at $5 = $6
    assert cost_usd(1_000_000, 1_000_000, "claude-haiku-4-5-20251001") == pytest.approx(6.0)


def test_cost_unknown_model_is_none():
    assert cost_usd(100, 100, "some-unknown-model") is None


def test_complete_returns_text_tokens_and_latency():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    result = complete(fake_client(response), "claude-haiku-4-5-20251001", "sys", "hi")
    assert result.text == "hello"
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert result.latency_ms >= 0


def test_complete_wraps_provider_errors():
    request = httpx.Request("POST", "https://example.invalid")
    client = fake_client(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError):
        complete(client, "claude-haiku-4-5-20251001", "sys", "hi")