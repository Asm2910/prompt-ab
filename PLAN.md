# Prompt A/B Platform: build plan

Request flow: POST /v1/completions {prompt_id, user_id, variables}
registry -> experiment running? -> splitter (hash of user_id) or active version
-> fill template -> call LLM -> log to Postgres -> return response
-> background: judge scores output -> stats compare variants -> dashboard.

## Phases
0. Warm-up: scripts/warmup.py. Call the LLM, request JSON, validate with Pydantic, print tokens and cost, add a judge returning reasoning plus a 1-5 score.
1. Registry: tables prompts, prompt_versions, active_versions, audit_log. Endpoints to create a prompt, add a version, list versions, activate a version (rollback = activate an older one). Template variables like {{context}} validated at request time.
2. Serving endpoint: POST /v1/completions end to end, no experiments yet.
3. Experiments and traffic splitting: experiments and variants tables, deterministic hashing (sha256 of experiment_id:user_id), assignment log.
4. Metrics: latency, tokens, cost, errors, plus an async judge score. The judge model must differ from the generator model.
5. Statistics: Welch t-test, Mann-Whitney U, confidence interval, planned sample size. No winner before the planned sample size (no peeking).
6. Dashboard: Streamlit, dark theme, fixed colour per variant, audit log page.
7. Demo: support-email classifier, 3 variants, ~500 simulated requests with a fixed random seed. pytest tests, README, 90-second video.

## Layout
app/ (main.py models.py registry.py splitter.py llm.py metrics.py stats.py)
dashboard/app.py, scripts/ (warmup.py simulate_traffic.py seed.py), tests/
docker-compose.yml, .env.example, README.md
