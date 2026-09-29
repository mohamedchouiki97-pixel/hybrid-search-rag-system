"""Before/after table from two evaluation runs (reads the *.results.json files).

    uv run python scripts/compare_runs.py reports/eval/baseline reports/eval/final > reports/eval/RESULTS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = [
    ("correctness", "Correctness"),
    ("faithfulness", "Faithfulness"),
    ("recall_at_k", "Recall@5"),
    ("mrr", "MRR"),
    ("citation_accuracy", "Citation acc."),
]
VARIANTS = ["fixed", "recursive", "semantic", "hybrid", "dense"]


def load(run: Path, variant: str) -> dict | None:
    path = run / f"{variant}.results.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.exists() else None


def cell(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    delta = after - before
    sign = "+" if delta > 0 else ""
    return f"{before:.2f} → **{after:.2f}** ({sign}{delta:.2f})" if abs(delta) >= 0.005 else f"{after:.2f} (=)"


def main(before_dir: str, after_dir: str) -> None:
    before_run, after_run = Path(before_dir), Path(after_dir)
    out = [f"# Evaluation: {before_run.name} → {after_run.name}", "",
           "50 golden questions, gpt-4.1-mini-2025-04-14 for generation and judging. "
           "fixed / recursive / semantic use hybrid retrieval; hybrid / dense use recursive chunks.", ""]  # fmt: skip
    header = "| Variant | " + " | ".join(label for _, label in METRICS) + " | Abstain recall | Abstain precision |"
    out += [header, "|---" * (len(METRICS) + 3) + "|"]
    for v in VARIANTS:
        b, a = load(before_run, v), load(after_run, v)
        if not (a and b):
            continue
        cells = [cell(b["overall"][m], a["overall"][m]) for m, _ in METRICS]
        cells += [cell(b["abstention"]["recall"], a["abstention"]["recall"]),
                  cell(b["abstention"]["precision"], a["abstention"]["precision"])]  # fmt: skip
        out.append(f"| {v} | " + " | ".join(cells) + " |")

    b, a = load(before_run, "recursive"), load(after_run, "recursive")
    if a and b:
        out += ["", "## By question type (recursive, hybrid)", "",
                "| Type | n | Correctness | Faithfulness | Recall@5 | Citation acc. |", "|---|---|---|---|---|---|"]  # fmt: skip
        for t, block in a["by_type"].items():
            bb = b["by_type"][t]
            out.append(
                f"| {t} | {block['n']} | {cell(bb['correctness'], block['correctness'])} | "
                f"{cell(bb['faithfulness'], block['faithfulness'])} | {cell(bb['recall_at_k'], block['recall_at_k'])} | "
                f"{cell(bb['citation_accuracy'], block['citation_accuracy'])} |"
            )
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
