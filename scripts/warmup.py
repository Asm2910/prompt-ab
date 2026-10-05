"""Phase 0 warm-up: classify support emails, validate JSON, then judge the result."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Verify these USD-per-million-token prices on the Anthropic pricing page:
# https://platform.claude.com/docs/en/about-claude/pricing
GENERATOR_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-5-5"
# Tuple is (input_usd_per_mtok, output_usd_per_mtok) for standard API calls.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    GENERATOR_MODEL: (1.0, 5.0),
    JUDGE_MODEL: (2.0, 10.0),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Classification(BaseModel):
    category: Literal["billing", "technical", "account", "general"]
    summary: str


class JudgeResult(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    input_price, output_price = PRICE_PER_MTOK[model]
    return (input_tokens / 1_000_000) * input_price + (
        output_tokens / 1_000_000
    ) * output_price


def _load_client() -> Anthropic:
    # Load .env from the repo root so the script works from any cwd.
    # The key never lives in source; missing values fail fast instead of calling the API empty.
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; copy .env.example to .env")
    return Anthropic(api_key=api_key)


def _extract_json_object(text: str) -> dict[str, object]:
    # Models sometimes wrap JSON in markdown fences or prose.
    # Slice from the first brace to the last so Pydantic sees a raw object.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    parsed: object = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload is not an object")
    return parsed


def _complete(
    client: Anthropic, model: str, system: str, user: str
) -> tuple[str, int, int, float]:
    # One Messages API round-trip: time wall-clock latency and read token usage
    # from the response so cost can be computed without a second billing API.
    started = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_s = time.perf_counter() - started
    text = "".join(block.text for block in message.content if block.type == "text")
    return text, message.usage.input_tokens, message.usage.output_tokens, latency_s


def classify(
    email: str, model: str, client: Anthropic
) -> tuple[Classification, int, int, float]:
    """Ask the generator for JSON only; retry once if parse or schema validation fails."""
    system = (
        "You classify customer support emails. "
        "Reply with JSON only, no markdown, in this shape: "
        '{"category": "billing"|"technical"|"account"|"general", "summary": "<one sentence>"}.'
    )
    user = f"Classify this email:\n\n{email}"
    input_tokens = 0
    output_tokens = 0
    latency_s = 0.0
    last_error: Exception | None = None

    # First attempt plus one retry. Token and latency totals include both calls
    # so a validation miss is visible in cost, not hidden by keeping only the success.
    for attempt in range(2):
        prompt = user
        if attempt == 1 and last_error is not None:
            prompt = (
                f"{user}\n\nYour previous reply was invalid ({last_error}). "
                "Reply with JSON only matching the schema."
            )
        text, in_tok, out_tok, call_latency = _complete(client, model, system, prompt)
        input_tokens += in_tok
        output_tokens += out_tok
        latency_s += call_latency
        try:
            return (
                Classification.model_validate(_extract_json_object(text)),
                input_tokens,
                output_tokens,
                latency_s,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"classification failed after retry: {last_error}")


def judge(
    email: str, result: Classification, model: str, client: Anthropic
) -> tuple[JudgeResult, int, int, float]:
    """Rate category + summary 1-5. Reasoning is requested first, score second."""
    system = (
        "You judge a support-email classification. "
        "Reply with JSON only, no markdown, fields in this order: "
        '{"reasoning": "<why the score>", "score": <integer 1-5>}. '
        "Write reasoning first, then the score. 1 is poor, 5 is excellent."
    )
    user = (
        f"Email:\n{email}\n\n"
        f"Classification category: {result.category}\n"
        f"Classification summary: {result.summary}"
    )
    input_tokens = 0
    output_tokens = 0
    latency_s = 0.0
    last_error: Exception | None = None

    # First attempt plus one retry. Token and latency totals include both calls
    # so a validation miss is visible in cost, not hidden by keeping only the success.
    for attempt in range(2):
        prompt = user
        if attempt == 1 and last_error is not None:
            prompt = (
                f"{user}\n\nYour previous reply was invalid ({last_error}). "
                "Reply with JSON only matching the schema."
            )
        text, in_tok, out_tok, call_latency = _complete(client, model, system, prompt)
        input_tokens += in_tok
        output_tokens += out_tok
        latency_s += call_latency
        try:
            return (
                JudgeResult.model_validate(_extract_json_object(text)),
                input_tokens,
                output_tokens,
                latency_s,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"judge failed after retry: {last_error}")


def main() -> None:
    # One client for the whole run. classify/judge reuse it instead of reloading .env.
    client = _load_client()

    # Five fixed emails so the warm-up is repeatable: one per category, then a
    # mixed billing/login/outage note that could reasonably land in more than one bucket.
    emails = [
        "Hi, my last invoice was charged twice for the same Pro plan. Please refund the duplicate $49 charge from March 3.",
        "The dashboard crashes with a 500 every time I export a CSV of last week's tickets. Stack trace mentions TimeoutError in reports.py.",
        "I cannot log in after resetting my password. The reset email arrived, but the new password is rejected and I am locked out of account #48219.",
        "What are your support hours for the Singapore region, and do you offer a phone line or email only?",
        "The payment failed during checkout and now I cannot log in after yesterday's outage. Not sure if this is billing or my account.",
    ]

    header = (
        f"{'excerpt':<42} {'category':<12} {'summary':<48} "
        f"{'score':>5} {'tok':>6} {'cost':>10} {'lat_s':>7}"
    )
    print(header)
    print("-" * len(header))

    generator_cost = 0.0
    judge_cost = 0.0
    for email in emails:
        classification, in_tok, out_tok, latency_s = classify(
            email, GENERATOR_MODEL, client
        )
        # Judge must use a different model than the generator so the score is not
        # self-graded. Row tokens, cost, and latency come from classify(); judge
        # cost is accumulated separately and printed after the table.
        judgment, judge_in, judge_out, _judge_latency = judge(
            email, classification, JUDGE_MODEL, client
        )

        row_tokens = in_tok + out_tok
        row_cost = cost_usd(in_tok, out_tok, GENERATOR_MODEL)
        generator_cost += row_cost
        judge_cost += cost_usd(judge_in, judge_out, JUDGE_MODEL)

        excerpt = email[:39] + "…" if len(email) > 40 else email
        summary = (
            classification.summary[:45] + "…"
            if len(classification.summary) > 46
            else classification.summary
        )
        print(
            f"{excerpt:<42} {classification.category:<12} {summary:<48} "
            f"{judgment.score:>5} {row_tokens:>6} ${row_cost:>8.6f} {latency_s:>7.2f}"
        )

    print(f"\nGenerator cost: ${generator_cost:.6f}")
    print(f"Judge cost: ${judge_cost:.6f}")
    print(f"Total cost: ${generator_cost + judge_cost:.6f}")


if __name__ == "__main__":
    main()
