import json

import pytest

from rag.core.fakes import FakeLLM
from rag.core.models import Answer, ChunkStrategy, Citation, Confidence, RetrievedChunk
from rag.evaluation.chunking_report import compare, run_chunking_comparison
from rag.evaluation.golden import GoldenItem
from rag.evaluation.runner import EvalRunner, summarize

GOOD = '{"score": 1, "supported_claims": 1, "total_claims": 1}'


@pytest.fixture
def items():
    def gi(i, qtype, sections):
        return GoldenItem.model_validate({"id": f"q{i}", "type": qtype, "question": f"Q{i}?", "gold_answer": "A.",
                                          "gold_chunk_sections": sections})  # fmt: skip

    gold = [{"doc_id": "d1", "section": "S1"}]
    return [gi(1, "lookup", gold), gi(2, "multi_hop", gold), gi(3, "no_answer", []), gi(4, "ambiguous", gold), gi(5, "no_answer", [])]


class StubPipeline:
    """Canned answers: answers q1, q2, q4 from d1|S1; abstains on q3; answers (wrongly) q5; q6 raises."""

    def __init__(self, make_chunk):
        self.hit = RetrievedChunk(chunk=make_chunk("fact", doc_id="d1", section_heading="S1"), score=0.9)
        self.asked = []

    def ask(self, question):
        self.asked.append(question)
        if question == "Q3?":
            return Answer(question=question, answer_text="I don't know.", abstained=True, retrieved=[self.hit])
        if question == "Q6?":
            raise TimeoutError("backend down")
        cite = Citation(marker=1, chunk_id=self.hit.chunk.chunk_id, claim_text="Fact.", verified=True)
        return Answer(question=question, answer_text="Fact [1].", citations=[cite], retrieved=[self.hit],
                      confidence=Confidence(composite=0.8))  # fmt: skip


def test_runner_end_to_end_with_stub_pipeline(tmp_path, items, make_chunk):
    pipeline = StubPipeline(make_chunk)
    summary = EvalRunner(FakeLLM(default=GOOD), k=5).run(pipeline, items, tmp_path, label="stub")

    assert pipeline.asked == [i.question for i in items]
    overall = summary["overall"]
    assert overall["n"] == 5
    assert overall["correctness"] == pytest.approx(4 / 5)  # q5 answered an unanswerable question
    assert overall["recall_at_k"] == 1.0 and overall["mrr"] == 1.0  # only questions with gold sections
    assert overall["citation_accuracy"] == 1.0
    assert summary["abstention"] == {"abstained": 1, "precision": 1.0, "recall": 0.5}
    assert summary["by_type"]["no_answer"]["correctness"] == 0.5

    data = json.loads((tmp_path / "stub.results.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in data["results"]] == ["q1", "q2", "q3", "q4", "q5"]
    assert data["results"][0]["retrieved"] == ["d1 | S1"]
    assert [r["retrieval_confidence"] for r in data["results"]][:2] == [0.0, 0.0]  # stub leaves confidence at 0
    md = (tmp_path / "stub.summary.md").read_text(encoding="utf-8")
    assert "| **overall** | 5 | 0.80 |" in md
    assert "Abstention precision 1.00, recall on no-answer questions 0.50." in md


def test_runner_records_pipeline_errors(tmp_path, make_chunk):
    item = GoldenItem(id="q6", type="lookup", question="Q6?", gold_answer="A.",
                      gold_chunk_sections=[{"doc_id": "d1", "section": "S1"}])  # fmt: skip
    summary = EvalRunner(FakeLLM(default=GOOD)).run(StubPipeline(make_chunk), [item], tmp_path, "err")
    assert summary["errors"] == 1 and summary["overall"]["correctness"] == 0.0
    result = json.loads((tmp_path / "err.results.json").read_text(encoding="utf-8"))["results"][0]
    assert result["error"] == "TimeoutError: backend down"


def test_summarize_ignores_missing_metrics():
    assert summarize([])["overall"]["correctness"] is None


def test_comparison_table(tmp_path, items, make_chunk):
    runner = EvalRunner(FakeLLM(default=GOOD))
    built = []

    def factory(strategy):
        built.append(strategy)
        return StubPipeline(make_chunk)

    summaries = run_chunking_comparison(runner, factory, items, tmp_path)
    assert built == list(ChunkStrategy)
    assert set(summaries) == {"fixed", "recursive", "semantic"}
    table = (tmp_path / "chunking_comparison.md").read_text(encoding="utf-8")
    for name in ("fixed", "recursive", "semantic"):
        assert f"| {name} | 0.80 |" in table
        assert (tmp_path / f"{name}.results.json").exists()


def test_generic_compare_for_hybrid_vs_dense(tmp_path, items, make_chunk):
    compare(EvalRunner(FakeLLM(default=GOOD)), ["hybrid", "dense"], lambda _m: StubPipeline(make_chunk), items,
            tmp_path, title="Hybrid vs dense", filename="retrieval_comparison.md")  # fmt: skip
    table = (tmp_path / "retrieval_comparison.md").read_text(encoding="utf-8")
    assert table.startswith("# Hybrid vs dense") and "| dense |" in table


class FlakyJudge:
    """Fails on the n-th call, like a dropped connection."""

    def __init__(self, fail_on):
        self.calls, self.fail_on = 0, fail_on

    def complete(self, system, user):
        self.calls += 1
        if self.calls == self.fail_on:
            raise ConnectionError("network down")
        return GOOD


def test_judge_failure_is_recorded_and_the_run_continues(tmp_path, items, make_chunk):
    summary = EvalRunner(FlakyJudge(fail_on=1)).run(StubPipeline(make_chunk), items, tmp_path, "flaky")
    assert summary["errors"] == 1 and summary["overall"]["n"] == 5
    results = json.loads((tmp_path / "flaky.results.json").read_text(encoding="utf-8"))["results"]
    assert results[0]["error"] == "judge failed: ConnectionError: network down"
    assert results[0]["correctness"] is None  # left out of averages instead of counted as 0
    assert all(r["error"] is None for r in results[1:])


def test_resume_reuses_clean_results_and_reruns_broken_ones(tmp_path, items, make_chunk):
    EvalRunner(FakeLLM(default=GOOD)).run(StubPipeline(make_chunk), items, tmp_path, "clean")
    EvalRunner(FlakyJudge(fail_on=1)).run(StubPipeline(make_chunk), items, tmp_path, "broken")

    runner = EvalRunner(FakeLLM(default=GOOD), resume=True)
    fresh = StubPipeline(make_chunk)
    assert runner.run(fresh, items, tmp_path, "clean")["errors"] == 0
    assert fresh.asked == []  # clean file reused, pipeline never called
    assert runner.run(fresh, items, tmp_path, "broken")["errors"] == 0
    assert len(fresh.asked) == 5  # file with errors was re-run
    assert runner.run(StubPipeline(make_chunk), items[:2], tmp_path, "clean")["overall"]["n"] == 2  # other questions: re-run
