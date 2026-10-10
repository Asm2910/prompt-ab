"""Send simulated support emails through the platform. Also used for the Phase 7 demo run.

Run (server must be running):
  uv run python scripts/simulate_traffic.py --prompt-id 1 --users 150 --prefix r1
"""

from __future__ import annotations

import argparse
import random

import httpx

EMAILS = [
    "I was charged twice for my Pro plan this month. Please refund the duplicate.",
    "The CSV export crashes the dashboard with a 500 error every time.",
    "I can't log in after resetting my password and I'm locked out.",
    "What are your support hours for the Singapore region?",
    "payment failed at checkout and now i cant login either, billing or account??",
    "refund",
    "How do I change the email address on my account? Also my last invoice looks wrong.",
    "The API returns 429 even though we are far below the documented rate limit.",
    "Do you offer an annual discount, and can I switch plans mid-cycle?",
    "my card got declined but the money left my bank account",
    "Thanks for the quick help yesterday, the issue is fixed!",
    "Sign-in with Google keeps looping back to the login page on Safari.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix", default="sim", help="user ids look like <prefix>-<n>")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)  # fixed seed: the same run can be repeated
    ok = failed = 0
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        for i in range(args.users):
            body = {
                "prompt_id": args.prompt_id,
                "user_id": f"{args.prefix}-{i}",
                "variables": {},
                "input": rng.choice(EMAILS),
            }
            try:
                response = client.post("/v1/completions", json=body)
                response.raise_for_status()
                ok += 1
            except httpx.HTTPError as exc:
                failed += 1
                print(f"request {i} failed: {exc}")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{args.users} sent ({ok} ok, {failed} failed)")
    print(f"done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()