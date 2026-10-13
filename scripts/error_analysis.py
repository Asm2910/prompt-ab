"""Where does the classifier go wrong? Accuracy by category, difficulty and variant, plus a check for
near-duplicate emails in the data file.

  uv run --env-file .env python scripts/error_analysis.py --experiment-id 3
  uv run python scripts/error_analysis.py --duplicates-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from sqlalchemy import create_engine, text

DEFAULT_EMAILS = Path(__file__).resolve().parent.parent / "data" / "support_emails.json"
CATEGORIES = ["account", "billing", "general", "technical"]
DIFFICULTIES = ["easy", "short", "hard"]


def extract_category(output: str) -> str | None:
    """Category from the LAST JSON object in a reply (chain-of-thought replies reason first)."""
    for block in reversed(re.findall(r"\{[^{}]*\}", output)):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        value = data.get("category") if isinstance(data, dict) else None
        if isinstance(value, str):
            return value.strip().lower()
    return None


def pct(correct: int, n: int) -> str:
    return f"{correct / n:6.1%} ({correct}/{n})" if n else "     -"


def load_rows(experiment_id: int, emails_by_text: dict) -> list[dict]:
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        labels = dict(conn.execute(
            text("select version_id, label from experiment_variants where experiment_id = :e"),
            {"e": experiment_id},
        ).all())
        raw = conn.execute(
            text("select input_text, output_text, variant_version_id from request_logs "
                 "where experiment_id = :e and error is null and output_text is not null"),
            {"e": experiment_id},
        ).all()
    rows = []
    for input_text, output_text, version_id in raw:
        known = emails_by_text.get(input_text)
        if known is None:
            continue
        predicted = extract_category(output_text)
        rows.append({
            "email_id": known["id"],
            "variant": labels.get(version_id, str(version_id)),
            "expected": known["category"],
            "difficulty": known["difficulty"],
            "predicted": predicted,
            "correct": predicted == known["category"],
        })
    print(f"{len(raw)} requests, {len(rows)} matched to the email file, "
          f"overall accuracy {sum(r['correct'] for r in rows) / len(rows):.1%}\n")
    return rows


def table(title: str, rows: list[dict], row_key, col_key, row_order, col_order) -> None:
    print(title)
    print(f"{'':14s}" + "".join(f"{c:>22s}" for c in col_order))
    for r in row_order:
        cells = []
        for c in col_order:
            group = [x for x in rows if row_key(x) == r and col_key(x) == c]
            cells.append(pct(sum(x["correct"] for x in group), len(group)))
        print(f"{r:14s}" + "".join(f"{cell:>22s}" for cell in cells))
    print()


def error_analysis(rows: list[dict]) -> None:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    table("Accuracy by true category and variant:", rows, lambda x: x["expected"], lambda x: x["variant"], CATEGORIES, variants)
    table("Accuracy by true category and difficulty (all variants):", rows, lambda x: x["expected"], lambda x: x["difficulty"], CATEGORIES, DIFFICULTIES)
    general = [r for r in rows if r["expected"] == "general"]
    table("True-general emails only: difficulty by variant:", general, lambda x: x["difficulty"], lambda x: x["variant"], DIFFICULTIES, variants)
    print("True-general emails the classifier got wrong: what it predicted, by difficulty")
    for d in DIFFICULTIES:
        wrong = Counter(r["predicted"] for r in general if r["difficulty"] == d and not r["correct"])
        total = sum(1 for r in general if r["difficulty"] == d)
        print(f"  {d:6s} {sum(wrong.values()):3d} wrong of {total:3d}   {dict(wrong.most_common())}")
    print()


def words(value: str) -> set[str]:
    return set(re.findall(r"[a-z']+", value.lower()))


def duplicate_pairs(emails: list[dict], threshold: float = 0.55) -> list[tuple[float, dict, dict]]:
    """Pairs of emails in the same category that share most of their words (Jaccard similarity)."""
    pairs = []
    by_category = defaultdict(list)
    for e in emails:
        by_category[e["category"]].append(e)
    for items in by_category.values():
        sets = [(e, words(e["text"])) for e in items]
        for (a, wa), (b, wb) in combinations(sets, 2):
            if wa and wb:
                similarity = len(wa & wb) / len(wa | wb)
                if similarity >= threshold:
                    pairs.append((similarity, a, b))
    return pairs


def representatives(emails: list[dict], pairs: list) -> set:
    """One email id per cluster of near-duplicates (the lowest id), plus every email with no twin."""
    parent = {e["id"]: e["id"] for e in emails}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a, b in pairs:
        root_a, root_b = find(a["id"]), find(b["id"])
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return {e["id"] for e in emails if find(e["id"]) == e["id"]}


def near_duplicates(emails: list[dict], pairs: list, threshold: float = 0.55) -> None:
    involved = {e["id"] for _, a, b in pairs for e in (a, b)}
    print(f"Near-duplicate check (same category, word overlap >= {threshold:.0%}): "
          f"{len(pairs)} pairs, {len(involved)} of {len(emails)} emails involved ({len(involved) / len(emails):.0%})")
    for similarity, a, b in sorted(pairs, key=lambda p: -p[0])[:8]:
        print(f"  {similarity:.0%}  [{a['id']}] {a['text'][:70]!r}\n        [{b['id']}] {b['text'][:70]!r}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--emails", type=Path, default=DEFAULT_EMAILS)
    parser.add_argument("--duplicates-only", action="store_true")
    parser.add_argument("--dedupe", action="store_true",
                        help="keep one email per near-duplicate cluster before analysing (a robustness check)")
    args = parser.parse_args()
    emails = json.loads(args.emails.read_text())
    pairs = duplicate_pairs(emails)
    if not args.duplicates_only:
        if args.experiment_id is None:
            sys.exit("Give --experiment-id, or use --duplicates-only.")
        rows = load_rows(args.experiment_id, {e["text"]: e for e in emails})
        if args.dedupe:
            keep = representatives(emails, pairs)
            rows = [r for r in rows if r["email_id"] in keep]
            print(f"--dedupe: kept {len(keep)} of {len(emails)} emails, so {len(rows)} requests; "
                  f"accuracy on them {sum(r['correct'] for r in rows) / len(rows):.1%}\n")
        error_analysis(rows)
    near_duplicates(emails, pairs)


if __name__ == "__main__":
    main()