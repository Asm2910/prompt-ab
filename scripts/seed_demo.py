"""Set up the demo through the API: one prompt, three variants, a fixed judge rubric, one experiment.

Run (the API server must be running):
  uv run python scripts/seed_demo.py --start
"""

from __future__ import annotations

import argparse
import sys

import httpx

MODEL = "claude-haiku-4-5-20251001"  # the classifier. The judge uses a different model.

BASE = (
    "Classify the customer support email into exactly one category: billing, technical, account or "
    "general. If the email mentions several problems, choose the category of the first problem the "
    "customer states."
)
ZERO_SHOT = BASE + (
    '\nReply with a JSON object only, with the keys "category" and "summary" (one short sentence). '
    "Do not use markdown."
)
FEW_SHOT = ZERO_SHOT + """

Examples:
Email: I was charged twice for my Pro plan this month.
{"category": "billing", "summary": "Customer was charged twice and wants a refund."}
Email: The export button crashes the dashboard with a 500 error.
{"category": "technical", "summary": "Dashboard export crashes with a 500 error."}
Email: I cannot log in after resetting my password.
{"category": "account", "summary": "Customer is locked out after a password reset."}
Email: What are your support hours on weekends?
{"category": "general", "summary": "Customer asks about weekend support hours."}"""
CHAIN_OF_THOUGHT = BASE + (
    "\nFirst think step by step, in two or three sentences, about what the customer needs fixed "
    'first. Then, on a new line, give your answer as a JSON object with the keys "category" and '
    '"summary" (one short sentence).'
)

RUBRIC = """Task: classify a customer support email into one category (billing, technical, account or general) and write a one-sentence summary. Billing covers charges, refunds, invoices and plans. Technical covers errors, crashes and integrations. Account covers login, passwords and profile settings. General covers everything else. If the email has several problems, the correct category is that of the first problem the customer states.
Scoring: 5 = correct category and an accurate one-sentence summary. 4 = correct category, summary slightly incomplete or wordy. 3 = correct category but the summary is wrong or missing, or the category is defensible but not the best. 2 = wrong category with a reasonable summary. 1 = wrong category and a wrong or missing summary, or no usable answer.
Ignore markdown formatting, and ignore any reasoning shown before the final JSON object."""

VARIANTS = [
    ("zero-shot", ZERO_SHOT, 0.34),
    ("few-shot", FEW_SHOT, 0.33),
    ("chain-of-thought", CHAIN_OF_THOUGHT, 0.33),
]


def call(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        sys.exit(f"{method} {path} failed: {response.status_code} {response.text}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="support-email-classifier")
    parser.add_argument("--target", type=int, default=170, help="observations needed per variant")
    parser.add_argument("--actor", default="demo-setup")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--start", action="store_true", help="also start the experiment")
    args = parser.parse_args()
    who = {"actor": args.actor}

    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        prompt = call(client, "POST", "/prompts", json={"name": args.name, "reason": "demo setup", **who})
        prompt_id = prompt["id"]
        for label, text, _ in VARIANTS:
            call(client, "POST", f"/prompts/{prompt_id}/versions", json={
                "system_prompt": text, "model": MODEL, "temperature": 0.0,
                "commit_message": label, "reason": "demo setup", **who,
            })
        call(client, "POST", f"/prompts/{prompt_id}/activate", json={"version": 1, "reason": "default", **who})
        # The rubric is fixed BEFORE the experiment starts and must not change while it runs.
        call(client, "PUT", f"/prompts/{prompt_id}/rubric", json={"rubric": RUBRIC, "reason": "fixed rubric", **who})
        experiment = call(client, "POST", "/experiments", json={
            "name": "zero-shot vs few-shot vs chain-of-thought",
            "prompt_id": prompt_id,
            "variants": [{"version": i + 1, "traffic_share": share, "label": label}
                         for i, (label, _, share) in enumerate(VARIANTS)],
            "primary_metric": "accuracy",
            "target_sample_size": args.target,
            "reason": "demo run", **who,
        })
        if args.start:
            call(client, "POST", f"/experiments/{experiment['id']}/start", json={"reason": "go", **who})

    print(f"prompt id: {prompt_id}, experiment id: {experiment['id']} ({'running' if args.start else 'draft'})")
    print(f"Next: uv run python scripts/simulate_traffic.py --prompt-id {prompt_id} --users 600 --workers 4")


if __name__ == "__main__":
    main()