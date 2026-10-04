# Prompt A/B

Platform for versioning prompts, serving completions, splitting traffic across variants, and comparing results with planned-sample statistics.

## Layout

```
app/            FastAPI service (main, models, registry, splitter, llm, metrics, stats)
dashboard/      Streamlit dashboard
scripts/        warmup, seed, and traffic simulation
tests/
```

## Setup

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Never commit secrets.

```bash
cp .env.example .env
docker compose up -d
```

Postgres 16 listens on `localhost:5432` by default (`DATABASE_URL` in `.env`).

## Phases (from PLAN.md)

0. Warm-up: LLM JSON + Pydantic + judge score
1. Prompt registry
2. Serving endpoint (`POST /v1/completions`)
3. Experiments and traffic splitting
4. Metrics and async judge
5. Statistics (no winner before planned sample size)
6. Streamlit dashboard
7. Demo (support-email classifier)
