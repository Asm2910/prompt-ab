"""Send simulated support emails through the platform, and score each answer against the known label.

Run (the API server must be running):
  uv run python scripts/simulate_traffic.py --prompt-id 1 --users 600 --workers 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

DEFAULT_EMAILS = Path(__file__).resolve().parent.parent / "data" / "support_emails.json"


def extract_category(text: str) -> str | None:
    """Category from the LAST JSON object in the reply (chain-of-thought replies reason first)."""
    for block in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        value = data.get("category") if isinstance(data, dict) else None
        if isinstance(value, str):
            return value.strip().lower()
    return None


def send_one(client: httpx.Client, prompt_id: int, user_id: str, email: dict) -> dict:
    """One request plus its accuracy score. Returns a small result record."""
    try:
        response = client.post(
            "/v1/completions",
            json={"prompt_id": prompt_id, "user_id": user_id, "variables": {}, "input": email["text"]},
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}

    result = {"ok": True, "version": body["version"], "correct": None}
    expected = email.get("category")
    if expected:
        predicted = extract_category(body["output"])
        # A reply with no readable category counts as wrong: a format failure is a failure.
        result["correct"] = predicted == expected
        try:
            client.post(
                f"/requests/{body['request_id']}/metrics",
                json={
                    "metric": "accuracy",
                    "value": 1.0 if result["correct"] else 0.0,
                    "note": f"expected {expected}, got {predicted}",
                },
            ).raise_for_status()
        except httpx.HTTPError as exc:
            result["error"] = f"accuracy not recorded: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--users", type=int, default=100, help="how many requests to send")
    parser.add_argument("--emails", type=Path, default=DEFAULT_EMAILS)
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix", default="sim", help="user ids look like <prefix>-<n>")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    emails = json.loads(args.emails.read_text())
    order = list(range(len(emails)))
    random.Random(args.seed).shuffle(order)  # fixed seed: the same run can be repeated
    if args.users > len(emails):
        print(f"Warning: {args.users} requests but only {len(emails)} distinct emails; some repeat.")

    ok = failed = 0
    tally: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # version -> [correct, scored]
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(send_one, client, args.prompt_id, f"{args.prefix}-{i}", emails[order[i % len(order)]])
                for i in range(args.users)
            ]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if not result["ok"]:
                    failed += 1
                    print(f"request failed: {result['error']}")
                    continue
                ok += 1
                if "error" in result:
                    print(result["error"])
                if result["correct"] is not None:
                    tally[result["version"]][0] += int(result["correct"])
                    tally[result["version"]][1] += 1
                if done % 50 == 0:
                    print(f"{done}/{args.users} done ({ok} ok, {failed} failed)")

    print(f"\ndone: {ok} ok, {failed} failed")
    for version in sorted(tally):
        correct, scored = tally[version]
        print(f"  version {version}: {correct}/{scored} correct ({correct / scored:.1%})")


if __name__ == "__main__":
    main()