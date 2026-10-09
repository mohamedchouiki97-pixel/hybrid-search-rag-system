import pytest

from rag.core.fakes import FakeLLM
from rag.core.models import Answer, Citation, RetrievedChunk
from rag.evaluation.golden import GoldenItem, GoldSection
from rag.evaluation.judge_cache import CachingLLMClient
from rag.evaluation.metrics import citation_accuracy, correctness, faithfulness, matches, mrr, recall_at_k

A, B, C = (
    GoldSection(doc_id="d1", section="A"),
    GoldSection(doc_id="d2", section="B"),
    GoldSection(doc_id="d3", section="C"),
)


@pytest.fixture
def ranked(make_chunk):
    """Retrieved order: d9|X, d1|A > sub, d2|B, d9|Y."""

    def rc(doc, heading, i):
        return RetrievedChunk(chunk=make_chunk(f"t{i}", doc_id=doc, chunk_index=i, section_heading=heading), score=1)

    return [rc("d9", "X", 0), rc("d1", "A > sub", 1), rc("d2", "B", 2), rc("d9", "Y", 3)]


def item(qtype, gold_answer="The port is 7420."):
    return GoldenItem(id="q1", type=qtype, question="Which port?", gold_answer=gold_answer)


def answer(text="Port 7420 [1].", abstained=False, citations=(), retrieved=()):
    return Answer(question="Which port?", answer_text=text, abstained=abstained,
                  citations=list(citations), retrieved=list(retrieved))  # fmt: skip


# ---------- retrieval ----------


def test_matches_section_and_subsections(make_chunk):
    def rc(doc, heading):
        return RetrievedChunk(chunk=make_chunk("t", doc_id=doc, section_heading=heading), score=1)

    assert matches(rc("d1", "A"), A)
    assert matches(rc("d1", "A > sub"), A)
    assert not matches(rc("d1", "AB"), A)  # prefix of a different section
    assert not matches(rc("d2", "A"), A)  # right section, wrong doc
    assert not matches(rc("d1", None), A)


def test_recall_at_k(ranked):
    assert recall_at_k(ranked, [A, B], k=3) == 1.0
    assert recall_at_k(ranked, [A, B], k=2) == 0.5
    assert recall_at_k(ranked, [A, B, C], k=5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, [C], k=5) == 0.0
    assert recall_at_k(ranked, [], k=5) is None


def test_mrr(ranked):
    assert mrr(ranked, [A]) == 0.5
    assert mrr(ranked, [B, A]) == 0.5  # first hit of any gold section
    assert mrr(ranked, [B]) == pytest.approx(1 / 3)
    assert mrr(ranked, [C]) == 0.0
    assert mrr([], [A]) == 0.0
    assert mrr(ranked, []) is None


# ---------- citations ----------


def test_citation_accuracy():
    def cite(v):
        return Citation(marker=1, chunk_id="c", claim_text="x", verified=v)

    assert citation_accuracy(answer(citations=[cite(True), cite(False), cite(True), cite(None)])) == 0.5
    assert citation_accuracy(answer()) is None


# ---------- correctness ----------


def test_no_answer_rewards_abstaining_and_penalises_answering():
    judge = FakeLLM(default='{"score": 1}')
    assert correctness(judge, item("no_answer"), answer(abstained=True)) == 1.0
    assert correctness(judge, item("no_answer"), answer("Use Kafka [1].")) == 0.0
    assert judge.call_count == 0  # decided without the judge


def test_abstaining_on_answerable_question_scores_zero():
    judge = FakeLLM(default='{"score": 1}')
    assert correctness(judge, item("lookup"), answer(abstained=True)) == 0.0
    assert judge.call_count == 0


@pytest.mark.parametrize(("reply", "expected"), [('{"score": 1}', 1.0), ('{"score": 0.5}', 0.5), ('{"score": 0}', 0.0),
                                                 ('{"score": 0.7}', None), ("nonsense", None)])  # fmt: skip
def test_correctness_judge_scores(reply, expected):
    assert correctness(FakeLLM(default=reply), item("lookup"), answer()) == expected


def test_correctness_prompt_contents_and_ambiguous_rule():
    judge = FakeLLM(default='{"score": 1}')
    correctness(judge, item("lookup"), answer("Port 7420 [1]."))
    correctness(judge, item("ambiguous"), answer("Did you mean X or Y?"))
    (sys1, user1), (sys2, _) = judge.calls
    assert "GOLD ANSWER:\nThe port is 7420." in user1 and "SYSTEM ANSWER:\nPort 7420 [1]." in user1
    assert "AMBIGUOUS" not in sys1 and "AMBIGUOUS" in sys2


# ---------- faithfulness ----------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [('{"supported_claims": 3, "total_claims": 4}', 0.75), ('{"supported_claims": 0, "total_claims": 0}', 1.0),
     ('{"supported_claims": 5, "total_claims": 4}', 1.0), ('{"supported_claims": "3", "total_claims": 4}', None),
     ('{"supported_claims": -1, "total_claims": 4}', None), ("oops", None)],
)  # fmt: skip
def test_faithfulness(reply, expected, make_chunk):
    ctx = [RetrievedChunk(chunk=make_chunk("The broker listens on port 7420."), score=1)]
    judge = FakeLLM(default=reply)
    assert faithfulness(judge, answer(retrieved=ctx)) == expected
    assert "CONTEXT:\n[1] The broker listens on port 7420." in judge.calls[0][1]


def test_faithfulness_skipped_when_abstained():
    judge = FakeLLM()
    assert faithfulness(judge, answer(abstained=True)) is None
    assert faithfulness(judge, answer(text="  ")) is None
    assert judge.call_count == 0


# ---------- judge cache ----------


def test_judge_cache_returns_stored_result_and_skips_llm(tmp_path):
    cache_file = tmp_path / "judge.sqlite"
    inner = FakeLLM(default="verdict-1")
    cached = CachingLLMClient(inner, cache_file, namespace="judge-model")
    assert cached.complete("s", "u") == "verdict-1"
    inner.default = "verdict-2"
    assert cached.complete("s", "u") == "verdict-1"  # stored, not recomputed
    assert inner.call_count == 1 and (cached.hits, cached.misses) == (1, 1)
    cached.close()

    reopened = CachingLLMClient(inner, cache_file, namespace="judge-model")
    assert reopened.complete("s", "u") == "verdict-1"  # persisted on disk
    assert inner.call_count == 1
    assert CachingLLMClient(inner, cache_file, namespace="other-model").complete("s", "u") == "verdict-2"
