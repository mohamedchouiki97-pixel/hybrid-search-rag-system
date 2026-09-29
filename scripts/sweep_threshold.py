"""Pick the abstain threshold from a calibration run.

    ABSTAIN_THRESHOLD=0 uv run python scripts/run_eval.py --what retrieval --out reports/eval/calibration
    uv run python scripts/sweep_threshold.py reports/eval/calibration/hybrid.results.json \
                                             reports/eval/calibration/dense.results.json

Writes a sweep table per file plus a combined one to reports/eval/threshold_sweep.md
and prints the recommended threshold (best for all files together).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.evaluation.threshold import best_threshold, render, sweep

OUT = Path("reports/eval/threshold_sweep.md")


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    sections, combined = ["# Abstain threshold sweep", ""], []
    for path in paths:
        results = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
        combined.extend(results)
        rows = sweep(results)
        sections.append(render(Path(path).stem.removesuffix(".results"), rows, best_threshold(rows)))
    rows = sweep(combined)
    chosen = best_threshold(rows)
    sections.append(render("combined", rows, chosen))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"Recommended abstain_threshold: {chosen:.2f} (written to {OUT})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
