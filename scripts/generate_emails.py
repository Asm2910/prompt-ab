"""Generate a labelled set of support emails for the demo run.

Each request asks Claude for emails whose PRIMARY issue belongs to one category, so the label is
known by construction. It is still machine-written, so audit a sample by hand.

  uv run python scripts/generate_emails.py                 # writes data/support_emails.json (about $0.50)
  uv run python scripts/generate_emails.py --audit 30      # prints 30 random emails to check by eye
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when this file is run directly

from app.llm import LLMError, complete, cost_usd, get_client  # noqa: E402

GENERATOR_MODEL = "claude-sonnet-5-5"
OUT_PATH = ROOT / "data" / "support_emails.json"

CATEGORIES = {
    "billing": "charges, refunds, invoices, payment methods, plans, pricing and discounts",
    "technical": "errors, crashes, bugs, slow performance, and API or integration problems",
    "account": "login, passwords, profile or email changes, permissions, deleting or securing an account",
    "general": "questions about hours, features or how to do something, plus feedback, thanks and partnerships",
}
PRODUCTS = [
    "a project management app", "an online store platform", "a video streaming service",
    "a banking app", "a fitness tracking app", "a travel booking site", "an email marketing tool",
    "a cloud storage service", "an accounting tool for freelancers", "a language learning app",
]
DIFFICULTIES = {"easy", "short", "hard"}
SYSTEM = "You write realistic customer support emails for testing a classifier. Reply with JSON only, no markdown."


def build_prompt(category: str, n: int, product: str) -> str:
    return (
        f"Write {n} different customer support emails sent to {product}. The PRIMARY issue of every "
        f"email must clearly belong to category '{category}' ({CATEGORIES[category]}).\n"
        "Mix the difficulty:\n"
        "- about 40% 'easy': clear, two to four sentences\n"
        "- about 30% 'short': under 12 words, casual, may have typos or no punctuation\n"
        "- about 30% 'hard': mentions a second, secondary issue from a different category AFTER the "
        "primary one, or is vague, but still has exactly one primary issue\n"
        "Vary names, tone, length and details.\n"
        'Reply with JSON only: {"emails": [{"text": "...", "difficulty": "easy|short|hard"}]}'
    )


def parse_emails(text: str) -> list[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    items = json.loads(text[start : end + 1]).get("emails")
    if not isinstance(items, list):
        raise ValueError("the reply has no emails list")
    cleaned = []
    for item in items:
        value = item.get("text") if isinstance(item, dict) else None
        if not isinstance(value, str) or not value.strip():
            continue
        difficulty = item.get("difficulty")
        cleaned.append({"text": value.strip(), "difficulty": difficulty if difficulty in DIFFICULTIES else "easy"})
    if not cleaned:
        raise ValueError("no usable emails")
    return cleaned


def generate_batch(client, category: str, n: int, product: str) -> tuple[list[dict], float]:
    prompt = build_prompt(category, n, product)
    cost = 0.0
    for _ in range(2):  # one retry on an unusable reply
        try:
            reply = complete(client, GENERATOR_MODEL, SYSTEM, prompt, max_tokens=3000)
        except LLMError as exc:
            print(f"  provider error ({category}): {exc}")
            continue
        cost += cost_usd(reply.input_tokens, reply.output_tokens, GENERATOR_MODEL) or 0.0
        try:
            return parse_emails(reply.text), cost
        except ValueError as exc:
            print(f"  unusable reply ({category}), retrying: {exc}")
    return [], cost


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    jobs = [(category, rng.choice(PRODUCTS)) for category in CATEGORIES for _ in range(args.batches)]
    try:
        client = get_client()
    except LLMError as exc:
        sys.exit(f"Cannot start: {exc}")

    collected: list[dict] = []
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(c, pool.submit(generate_batch, client, c, args.per_batch, p)) for c, p in jobs]
        for done, (category, future) in enumerate(futures, 1):
            emails, cost = future.result()
            total_cost += cost
            collected.extend({**e, "category": category} for e in emails)
            if done % 10 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} batches, {len(collected)} emails, ${total_cost:.3f} so far")

    seen: set[str] = set()
    unique = []
    for item in collected:
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    rng.shuffle(unique)
    for i, item in enumerate(unique, 1):
        item["id"] = i

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(unique, indent=1, ensure_ascii=False) + "\n")
    by_category = {c: sum(1 for e in unique if e["category"] == c) for c in CATEGORIES}
    print(f"Wrote {len(unique)} emails to {OUT_PATH} (cost ${total_cost:.3f}); per category: {by_category}")
    print("Next: audit a sample by hand with --audit 30")


def audit(args: argparse.Namespace) -> None:
    emails = json.loads(OUT_PATH.read_text())
    sample = random.Random(args.seed).sample(emails, min(args.audit, len(emails)))
    print("Check each label. The rule: the category of the FIRST problem the customer states.\n")
    for e in sample:
        print(f"[{e['id']}] {e['category']} / {e['difficulty']}\n    {e['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, default=15, help="batches per category")
    parser.add_argument("--per-batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--audit", type=int, default=0, help="print N random emails to check, then exit")
    arguments = parser.parse_args()
    audit(arguments) if arguments.audit else generate(arguments)