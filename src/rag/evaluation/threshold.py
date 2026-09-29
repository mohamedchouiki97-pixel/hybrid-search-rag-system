"""Choose the abstain threshold from evaluation data instead of guessing.

Needs results from a *calibration run* with ABSTAIN_THRESHOLD=0: every question then
reaches the model, so we know, per question, its retrieval confidence, whether the
model refused, and how correct its answer was. Replaying any threshold t is then free:

    abstains at t  <=>  retrieval_confidence < t  or  the model refused
    score at t     =    no_answer:  1 if it abstains else 0
                        others:     0 if it abstains else its recorded correctness

The best threshold maximizes mean score. When several thresholds tie, the middle of
the best range is chosen: it leaves the most margin on both sides for questions we
have not seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

GRID = [round(i * 0.05, 2) for i in range(20)]  # 0.00 .. 0.95


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    score: float  # mean correctness if this threshold had been used
    abstain_recall: float | None  # share of no_answer questions abstained on
    false_abstains: int  # answerable questions it would refuse


def replay(results: Sequence[dict], threshold: float) -> SweepRow:
    scores, no_answer_hits, no_answer_total, false_abstains = [], 0, 0, 0
    for r in results:
        conf = r["retrieval_confidence"] or 0.0
        abstains = conf < threshold or r["abstained"]
        if r["type"] == "no_answer":
            no_answer_total += 1
            no_answer_hits += abstains
            scores.append(1.0 if abstains else 0.0)
        else:
            false_abstains += abstains
            scores.append(0.0 if abstains else (r["correctness"] or 0.0))
    recall = no_answer_hits / no_answer_total if no_answer_total else None
    return SweepRow(threshold, fmean(scores) if scores else 0.0, recall, false_abstains)


def sweep(results: Sequence[dict], grid: Sequence[float] = GRID) -> list[SweepRow]:
    return [replay(results, t) for t in grid]


def best_threshold(rows: Sequence[SweepRow]) -> float:
    """Middle of the (first) contiguous run of thresholds with the top score."""
    top = max(r.score for r in rows)
    start = next(i for i, r in enumerate(rows) if r.score == top)
    end = start
    while end + 1 < len(rows) and rows[end + 1].score == top:
        end += 1
    return rows[(start + end) // 2].threshold


def render(label: str, rows: Sequence[SweepRow], chosen: float) -> str:
    lines = [f"## {label}", "", "| Threshold | Score | Abstain recall | False abstains |", "|---|---|---|---|"]
    for r in rows:
        mark = " **<- chosen**" if r.threshold == chosen else ""
        recall = "n/a" if r.abstain_recall is None else f"{r.abstain_recall:.2f}"
        lines.append(f"| {r.threshold:.2f}{mark} | {r.score:.3f} | {recall} | {r.false_abstains} |")
    return "\n".join(lines) + "\n"
