"""Thin wrapper around the Anthropic API: one call, tokens, latency, cost."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (input_usd_per_mtok, output_usd_per_mtok). Verify on the Anthropic pricing page:
# https://platform.claude.com/docs/en/about-claude/pricing
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}


class LLMError(Exception):
    """Raised when the provider call fails. The message never contains the API key."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float | None:
    """Dollar cost of one call, or None if we have no price for this model."""
    prices = PRICE_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1_000_000) * input_price + (
        output_tokens / 1_000_000
    ) * output_price


@lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """Build the client once, on first use. The key is read from .env, never from code."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set in .env")
    return Anthropic(api_key=api_key)


def complete(
    client: Anthropic, model: str, system: str, user: str, max_tokens: int = 1024
) -> LLMResult:
    """One Messages API round trip. Wall-clock latency, token usage read from the response."""
    started = time.perf_counter()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)
    text = "".join(block.text for block in message.content if block.type == "text")
    return LLMResult(
        text=text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        latency_ms=latency_ms,
    )