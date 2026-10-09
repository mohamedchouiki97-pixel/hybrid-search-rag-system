"""Evaluation runner: every golden question through a pipeline, scored, written to disk.

Outputs, in out_dir:
- <label>.results.json   one record per question plus the summary
- <label>.summary.md     overall and per-type averages, abstention quality
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Protocol

from rag.core.interfaces import LLMClient
from rag.core.models import Answer
from rag.evaluation.golden import GoldenItem, QuestionType
from rag.evaluation.metrics import citation_accuracy, correctness, faithfulness, mrr, recall_at_k

METRICS = ("correctness", "faithfulness", "recall_at_k", "mrr", "citation_accuracy")


class Pipeline(Protocol):
    def ask(self, question: str) -> Answer: ...


@dataclass
class QuestionResult:
    id: str
    type: str
    question: str
    answer_text: str = ""
    abstained: bool = False
    correctness: float | None = None
    faithfulness: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    citation_accuracy: float | None = None
    confidence: float | None = None
    retrieval_confidence: float | None = None  # what the abstain threshold is compared against
    retrieved: list[str] = field(default_factory=list)  # "doc_id | section" of each retrieved chunk
    latency_s: float = 0.0
    error: str | None = None


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return fmean(present) if present else None


def summarize(results: Sequence[QuestionResult]) -> dict:
    """Averages overall and per type, plus abstention precision / recall."""

    def block(rows: Sequence[QuestionResult]) -> dict:
        return {"n": len(rows), **{m: _mean([getattr(r, m) for r in rows]) for m in METRICS}}

    no_answer = [r for r in results if r.type == QuestionType.NO_ANSWER]
    abstained = [r for r in results if r.abstained]
    return {
        "overall": block(results),
        "by_type": {t.value: block([r for r in results if r.type == t]) for t in QuestionType},
        "abstention": {
            "abstained": len(abstained),
            # of the times it abstained, how often was that right?
            "precision": (sum(1 for r in abstained if r.type == QuestionType.NO_ANSWER) / len(abstained))
            if abstained
            else None,
            # of the unanswerable questions, how many did it abstain on?
            "recall": (sum(1 for r in no_answer if r.abstained) / len(no_answer)) if no_answer else None,
        },
        "errors": sum(1 for r in results if r.error),
        "mean_latency_s": _mean([r.latency_s for r in results]),
    }


class EvalRunner:
    """resume=True reuses a variant's existing results file when it covers the same
    questions and recorded zero errors, so an interrupted comparison only re-runs
    what is missing or broken."""

    def __init__(
        self, judge: LLMClient, k: int = 5, progress: Callable[[str], None] | None = None, resume: bool = False
    ) -> None:
        self.judge = judge
        self.k = k
        self.progress = progress or (lambda _msg: None)
        self.resume = resume

    def evaluate_one(self, pipeline: Pipeline, item: GoldenItem) -> QuestionResult:
        result = QuestionResult(id=item.id, type=item.type.value, question=item.question)
        start = time.perf_counter()
        try:
            answer = pipeline.ask(item.question)
        except Exception as exc:  # one broken question must not stop the run
            result.error = f"{type(exc).__name__}: {exc}"
            result.correctness = 0.0
            result.latency_s = time.perf_counter() - start
            return result
        result.latency_s = time.perf_counter() - start
        result.answer_text = answer.answer_text
        result.abstained = answer.abstained
        result.confidence = answer.confidence.composite
        result.retrieval_confidence = answer.confidence.retrieval
        result.retrieved = [f"{rc.chunk.doc_id} | {rc.chunk.section_heading}" for rc in answer.retrieved]
        result.recall_at_k = recall_at_k(answer.retrieved, item.gold_chunk_sections, self.k)
        result.mrr = mrr(answer.retrieved, item.gold_chunk_sections)
        result.citation_accuracy = citation_accuracy(answer)
        try:
            result.correctness = correctness(self.judge, item, answer)
            result.faithfulness = faithfulness(self.judge, answer)
        except Exception as exc:  # a failed judge call must not stop the run either
            result.error = f"judge failed: {type(exc).__name__}: {exc}"
        return result

    def _reusable(self, path: Path, items: Sequence[GoldenItem]) -> dict | None:
        if not (self.resume and path.exists()):
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        same_questions = [r["id"] for r in data["results"]] == [i.id for i in items]
        return data["summary"] if same_questions and data["summary"]["errors"] == 0 else None

    def run(self, pipeline: Pipeline, items: Sequence[GoldenItem], out_dir: str | Path, label: str) -> dict:
        reused = self._reusable(Path(out_dir) / f"{label}.results.json", items)
        if reused is not None:
            self.progress(f"[{label}] reusing existing results")
            return reused
        results = []
        for n, item in enumerate(items, start=1):
            results.append(self.evaluate_one(pipeline, item))
            self.progress(f"[{label}] {n}/{len(items)} {item.id}")
        summary = summarize(results)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"label": label, "k": self.k, "summary": summary, "results": [asdict(r) for r in results]}
        (out / f"{label}.results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / f"{label}.summary.md").write_text(render_summary(label, summary, self.k), encoding="utf-8")
        return summary


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_summary(label: str, summary: dict, k: int) -> str:
    header = f"| Type | n | Correctness | Faithfulness | Recall@{k} | MRR | Citation acc. |"
    lines = [f"# Evaluation: {label}", "", header, "|---|---|---|---|---|---|---|"]
    for name, block in [("**overall**", summary["overall"]), *summary["by_type"].items()]:
        lines.append(f"| {name} | {block['n']} | " + " | ".join(fmt(block[m]) for m in METRICS) + " |")
    a = summary["abstention"]
    lines += [
        "",
        f"Abstained on {a['abstained']} questions. "
        f"Abstention precision {fmt(a['precision'])}, recall on no-answer questions {fmt(a['recall'])}.",
        f"Errors: {summary['errors']}. Mean latency: {fmt(summary['mean_latency_s'])} s.",
        "",
    ]
    return "\n".join(lines)
