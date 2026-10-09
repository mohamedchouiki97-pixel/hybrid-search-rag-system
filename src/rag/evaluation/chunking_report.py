"""Side-by-side comparisons: one eval run per variant, then one table.

Used for the chunking comparison (fixed vs recursive vs semantic) and for
hybrid vs dense-only retrieval. The caller supplies a factory that builds the
pipeline for each variant (service.py does this with real modules).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from rag.core.models import ChunkStrategy
from rag.evaluation.golden import GoldenItem
from rag.evaluation.runner import METRICS, EvalRunner, Pipeline, fmt

LABELS = {
    "correctness": "Correctness",
    "faithfulness": "Faithfulness",
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "citation_accuracy": "Citation acc.",
}


def compare(
    runner: EvalRunner,
    variants: Sequence[str],
    make_pipeline: Callable[[str], Pipeline],
    items: Sequence[GoldenItem],
    out_dir: str | Path,
    title: str,
    filename: str,
) -> dict[str, dict]:
    """Run each variant, write <filename> with the comparison table, return summaries by variant."""
    summaries = {v: runner.run(make_pipeline(v), items, out_dir, label=v) for v in variants}
    header = "| Variant | " + " | ".join(LABELS[m] for m in METRICS) + " | Abstain recall | Latency (s) |"
    lines = [
        f"# {title}",
        "",
        f"{len(items)} questions, k = {runner.k}.",
        "",
        header,
        "|---" * (len(METRICS) + 3) + "|",
    ]
    for v, s in summaries.items():
        cells = [fmt(s["overall"][m]) for m in METRICS] + [fmt(s["abstention"]["recall"]), fmt(s["mean_latency_s"])]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    Path(out_dir, filename).write_text("\n".join(lines), encoding="utf-8")
    return summaries


def run_chunking_comparison(
    runner: EvalRunner,
    make_pipeline: Callable[[ChunkStrategy], Pipeline],
    items: Sequence[GoldenItem],
    out_dir: str | Path,
) -> dict[str, dict]:
    return compare(
        runner,
        [s.value for s in ChunkStrategy],
        lambda name: make_pipeline(ChunkStrategy(name)),
        items,
        out_dir,
        title="Chunking strategy comparison",
        filename="chunking_comparison.md",
    )
