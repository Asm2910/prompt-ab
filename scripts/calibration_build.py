"""Build a calibration set: outputs for YOU to score blind, plus the judge's scores for the same outputs.

Mixes real wrong answers, real right answers and deliberately corrupted ones from an experiment.
Run (the server need not be running, but .env must be loaded for the database):

  uv run --env-file .env python scripts/calibration_build.py --prompt-id 2 --experiment-id 3 --seed 1 --out calibration/round1

Then fill in the human_score column of <out>/to_label.csv WITHOUT looking at items.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when this file is run directly

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.calibration import Candidate, build_items  # noqa: E402
from app.llm import LLMError, get_client  # noqa: E402
from app.metrics import JudgeError, judge_output  # noqa: E402
from app.models import JudgeRubric, MetricScore, RequestLog, get_engine  # noqa: E402


def load_candidates(session: Session, experiment_id: int) -> list[Candidate]:
    rows = session.execute(
        select(RequestLog.id, RequestLog.input_text, RequestLog.output_text, MetricScore.value)
        .join(MetricScore, MetricScore.request_id == RequestLog.id)
        .where(
            RequestLog.experiment_id == experiment_id,
            MetricScore.metric == "accuracy",
            RequestLog.error.is_(None),
            RequestLog.output_text.is_not(None),
        )
    ).all()
    return [Candidate(rid, text_in, text_out, value == 1.0) for rid, text_in, text_out, value in rows]


def judge_all(client, rubric: str, items: list[dict], workers: int) -> float:
    """Score every item with the judge, using the same rubric the experiment used."""
    def score(item: dict):
        try:
            verdict, cost = judge_output(client, rubric, item["input"], item["output"])
            return verdict.score, verdict.reasoning, cost
        except JudgeError as exc:
            return None, f"judge failed: {exc}", exc.cost

    total = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item, (score_value, reasoning, cost) in zip(items, pool.map(score, items)):
            item["judge_score"], item["judge_reasoning"] = score_value, reasoning
            total += cost
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1, help="use a different seed for each round")
    parser.add_argument("--out", type=Path, default=ROOT / "calibration" / "round1")
    parser.add_argument("--wrong", type=int, default=8)
    parser.add_argument("--right", type=int, default=8)
    parser.add_argument("--corrupt", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with Session(get_engine()) as session:
        rubric_row = session.get(JudgeRubric, args.prompt_id)
        if rubric_row is None:
            sys.exit(f"Prompt {args.prompt_id} has no rubric. Set one with PUT /prompts/{{id}}/rubric.")
        rubric = rubric_row.rubric
        candidates = load_candidates(session, args.experiment_id)
    if not candidates:
        sys.exit(f"No scored requests found for experiment {args.experiment_id}.")
    wrong = sum(not c.correct for c in candidates)
    print(f"{len(candidates)} candidates ({wrong} wrong, {len(candidates) - wrong} right).")

    items = build_items(candidates, random.Random(args.seed), args.wrong, args.right, args.corrupt)
    try:
        cost = judge_all(get_client(), rubric, items, args.workers)
    except LLMError as exc:
        sys.exit(f"Cannot start the judge: {exc}")
    failed = sum(item["judge_score"] is None for item in items)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")
    (args.out / "rubric.txt").write_text(rubric + "\n")
    with (args.out / "to_label.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "input", "output", "human_score", "note"])
        for item in items:  # no kind and no judge score: you must not see them
            writer.writerow([item["id"], item["input"], item["output"], "", ""])

    print(f"Wrote {len(items)} items to {args.out} (judge cost ${cost:.3f}, {failed} judge failures).")
    print(f"1. Read {args.out / 'rubric.txt'}.")
    print(f"2. Fill in human_score (1-5) in {args.out / 'to_label.csv'}. Do not open items.json yet.")
    print(f"3. Run: uv run python scripts/calibration_report.py --dir {args.out}")


if __name__ == "__main__":
    main()