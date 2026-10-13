"""Compare your scores with the judge's scores and report how well they agree.

  uv run python scripts/calibration_report.py --dir calibration/round1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app` can be imported when this file is run directly

from app.calibration import SCORES, agreement_report  # noqa: E402


def fmt(value: float | None, digits: int = 2) -> str:
    return "undefined" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def load_human_scores(path: Path) -> dict[str, tuple[int, str]]:
    scores: dict[str, tuple[int, str]] = {}
    missing: list[str] = []
    # utf-8-sig: Excel and Numbers add an invisible marker at the start of a CSV file.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("human_score") or "").strip()
            if raw not in {str(s) for s in SCORES}:
                missing.append(row["id"])
                continue
            scores[row["id"]] = (int(raw), (row.get("note") or "").strip())
    if missing:
        sys.exit(f"{len(missing)} rows have no valid human_score (1-5): {', '.join(missing[:10])}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()

    items = json.loads((args.dir / "items.json").read_text())
    human = load_human_scores(args.dir / "to_label.csv")
    usable = [i for i in items if i["id"] in human and i["judge_score"] is not None]
    skipped = len(items) - len(usable)
    h = [human[i["id"]][0] for i in usable]
    j = [i["judge_score"] for i in usable]
    report = agreement_report(h, j)

    print(f"Items compared: {report['n']}" + (f" ({skipped} skipped: judge failed)" if skipped else ""))
    print(f"Exact agreement:           {report['exact_agreement']:.0%}")
    print(f"Within one point:          {report['within_one']:.0%}")
    print(f"Weighted kappa:            {fmt(report['weighted_kappa'])} "
          f"(95% interval {fmt(report['kappa_ci_low'])} to {fmt(report['kappa_ci_high'])}), "
          f"{report['kappa_band']}")
    print(f"Spearman correlation:      {fmt(report['spearman'])}")
    print(f"Judge minus human, mean:   {report['mean_bias']:+.2f}  (positive = the judge is more lenient)")

    print("\nRows are your score, columns the judge's score:")
    print("        " + "".join(f"{s:>5}" for s in SCORES))
    for score, row in zip(SCORES, report["confusion"]):
        print(f"human {score} " + "".join(f"{count:>5}" for count in row))

    print("\nBy kind of output (mean score):")
    for kind in sorted({i["kind"] for i in usable}):
        group = [i for i in usable if i["kind"] == kind]
        print(f"  {kind:28s} n={len(group):2d}  human {sum(human[i['id']][0] for i in group) / len(group):.2f}"
              f"   judge {sum(i['judge_score'] for i in group) / len(group):.2f}")

    big = sorted((i for i in usable if abs(i["judge_score"] - human[i["id"]][0]) >= 2),
                 key=lambda i: -abs(i["judge_score"] - human[i["id"]][0]))
    print(f"\nLargest disagreements (two or more points): {len(big)}")
    for i in big[:6]:
        score, note = human[i["id"]]
        print(f"\n[{i['id']}] {i['kind']}: human {score}, judge {i['judge_score']}")
        print(f"  email:  {i['input'][:160]}")
        print(f"  output: {i['output'][:160]!r}")
        print(f"  judge says: {i['judge_reasoning'][:240]}")
        if note:
            print(f"  your note: {note}")

    (args.dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nSaved {args.dir / 'report.json'}")


if __name__ == "__main__":
    main()