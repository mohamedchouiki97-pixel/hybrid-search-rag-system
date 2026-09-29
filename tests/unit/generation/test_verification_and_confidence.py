import pytest

from rag.core.config import ConfidenceWeights
from rag.core.fakes import FakeLLM
from rag.core.interfaces import CitationVerifier
from rag.core.models import Answer, Citation, RetrievedChunk
from rag.generation.citation_verifier import NOT_NEEDED_REASON, NOT_RETRIEVED_REASON, LLMCitationVerifier
from rag.generation.confidence import ConfidenceScorer, citation_coverage, composite, retrieval_confidence
from rag.generation.generator import UNMATCHED_REASON, parse_citations
from rag.generation.prompts import parse_json_object

YES = '{"supported": true, "reason": "stated directly"}'
NO = '{"supported": false, "reason": "passage is about backups"}'


@pytest.fixture
def chunks(make_chunk):
    texts = ["The broker listens on port 7420.", "Backups are incremental."]
    return [RetrievedChunk(chunk=make_chunk(t, doc_id=f"d{i}"), score=0.8, rerank_score=0.8) for i, t in enumerate(texts)]


def make_answer(text, chunks):
    return Answer(question="q", answer_text=text, citations=parse_citations(text, chunks), retrieved=chunks)


# ---------- JSON parsing ----------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (YES, {"supported": True, "reason": "stated directly"}),
        ('```json\n{"score": 0.5}\n```', {"score": 0.5}),
        ('Sure! {"supported": false} hope that helps', {"supported": False}),
        ("not json", None),
        ("{broken", None),
        ("[1, 2]", None),
        ("", None),
    ],
)
def test_parse_json_object(reply, expected):
    assert parse_json_object(reply) == expected


# ---------- verifier ----------


def test_verifier_marks_supported_and_unsupported(chunks):
    judge = FakeLLM({"CLAIM:\nPort is 7420.": YES, "CLAIM:\nBackups are full copies.": NO})
    answer = make_answer("Port is 7420 [1]. Backups are full copies [2].", chunks)
    checked = LLMCitationVerifier(judge).verify(answer, chunks)
    assert [(c.verified, c.judge_reason) for c in checked.citations] == [
        (True, "stated directly"),
        (False, "passage is about backups"),
    ]
    assert judge.call_count == 2
    assert any("PASSAGES:\n[1] The broker listens on port 7420." in user for _, user in judge.calls)


def test_claim_is_judged_once_against_all_its_passages(chunks):
    judge = FakeLLM(default='{"supported": true, "used": [1, 2], "reason": "combined"}')
    answer = make_answer("The port is 7420 and backups are incremental [1][2].", chunks)
    checked = LLMCitationVerifier(judge).verify(answer, chunks)
    assert [c.verified for c in checked.citations] == [True, True]
    [(_, user)] = judge.calls  # one call for the claim, not one per citation
    assert "[1] The broker listens on port 7420." in user and "[2] Backups are incremental." in user


def test_cited_passage_the_judge_did_not_need_is_flagged(chunks):
    judge = FakeLLM(default='{"supported": true, "used": [1], "reason": "port stated"}')
    checked = LLMCitationVerifier(judge).verify(make_answer("The port is 7420 [1][2].", chunks), chunks)
    assert [(c.marker, c.verified) for c in checked.citations] == [(1, True), (2, False)]
    assert checked.citations[1].judge_reason == NOT_NEEDED_REASON


def test_unsupported_claim_marks_all_its_citations(chunks):
    judge = FakeLLM(default='{"supported": false, "used": [], "reason": "not stated"}')
    checked = LLMCitationVerifier(judge).verify(make_answer("The port is 9000 [1][2].", chunks), chunks)
    assert [(c.verified, c.judge_reason) for c in checked.citations] == [(False, "not stated")] * 2


@pytest.mark.parametrize("reply", ["garbage", '{"supported": "yes"}', '{"reason": "missing verdict"}'])
def test_malformed_judge_output_does_not_crash(chunks, reply):
    checked = LLMCitationVerifier(FakeLLM(default=reply)).verify(make_answer("Port is 7420 [1].", chunks), chunks)
    [c] = checked.citations
    assert c.verified is False and c.judge_reason.startswith("unparseable judge output")


def test_judge_exception_marks_unsupported(chunks):
    class Broken:
        def complete(self, system, user):
            raise TimeoutError("slow")

    [c] = LLMCitationVerifier(Broken()).verify(make_answer("Port is 7420 [1].", chunks), chunks).citations
    assert c.verified is False and "judge call failed" in c.judge_reason


def test_unmatched_and_foreign_citations_skip_the_judge(chunks):
    judge = FakeLLM(default=YES)
    answer = make_answer("Invented [9].", chunks)
    answer.citations.append(Citation(marker=1, chunk_id="not-retrieved", claim_text="x"))
    checked = LLMCitationVerifier(judge).verify(answer, chunks)
    assert [(c.verified, c.judge_reason) for c in checked.citations] == [
        (False, UNMATCHED_REASON),
        (False, NOT_RETRIEVED_REASON),
    ]
    assert judge.call_count == 0


def test_verifier_runs_concurrently_and_keeps_order(chunks):
    text = " ".join(f"Claim number {i} [1]." for i in range(20))
    judge = FakeLLM(default=YES)
    checked = LLMCitationVerifier(judge, max_workers=8).verify(make_answer(text, chunks), chunks)
    assert [c.claim_text for c in checked.citations] == [f"Claim number {i}." for i in range(20)]
    assert judge.call_count == 20


def test_no_citations_returns_answer_unchanged(chunks):
    answer = make_answer("Nothing cited.", chunks)
    assert LLMCitationVerifier(FakeLLM()).verify(answer, chunks) is answer


def test_verifier_satisfies_protocol():
    assert isinstance(LLMCitationVerifier(FakeLLM()), CitationVerifier)


# ---------- confidence ----------


def test_retrieval_confidence(make_chunk):
    def rc(score, rerank=None):
        return RetrievedChunk(chunk=make_chunk("t"), score=score, rerank_score=rerank)

    assert retrieval_confidence([]) == 0.0
    assert retrieval_confidence([rc(0.02, rerank=0.91), rc(0.5)]) == 0.91  # hybrid: rerank score
    assert retrieval_confidence([rc(0.64)]) == 0.64  # dense: cosine
    assert retrieval_confidence([rc(-0.2)]) == 0.0
    assert retrieval_confidence([rc(1.0, rerank=3.0)]) == 1.0
    # dense order is not rerank order: the best-scored chunk counts, wherever it is
    assert retrieval_confidence([rc(0.8, rerank=0.01), rc(0.7, rerank=0.02), rc(0.6, rerank=0.93)]) == 0.93


def verified(text, flags):
    cites = [Citation(marker=1, chunk_id="c", claim_text=t, verified=f) for t, f in zip(text, flags, strict=True)]
    return cites


@pytest.mark.parametrize(
    ("answer_text", "citations", "expected"),
    [
        ("A [1]. B [1].", [("A.", True), ("B.", True)], 1.0),
        ("A [1]. B [1].", [("A.", True), ("B.", False)], 0.5),
        ("A [1]. B. C. D [1].", [("A.", True), ("B. C. D.", True)], 1.0),  # B, C grouped with D
        ("A [1]. B. C.", [("A.", True)], 1 / 3),  # trailing uncited sentences count against
        ("A [1]. B. C. D [1].", [("A.", True), ("B. C. D.", False)], 0.25),  # coverage counts sentences
        ("A [1][2].", [("A.", False), ("A.", True)], 1.0),  # one verified citation is enough
        ("A [1].", [("A.", None)], 0.0),  # never verified
        ("Found:", [], 0.0),  # no claims at all
    ],
)
def test_citation_coverage(answer_text, citations, expected):
    cites = [Citation(marker=1, chunk_id="c", claim_text=t, verified=v) for t, v in citations]
    assert citation_coverage(Answer(question="q", answer_text=answer_text, citations=cites)) == pytest.approx(expected)


def test_coverage_is_zero_when_abstained():
    cites = [Citation(marker=1, chunk_id="c", claim_text="A.", verified=True)]
    assert citation_coverage(Answer(question="q", answer_text="A [1].", citations=cites, abstained=True)) == 0.0


def test_composite_matches_weights():
    w = ConfidenceWeights(retrieval=1, citation_coverage=1, completeness=2)
    assert composite(0.8, 0.4, 0.5, w) == pytest.approx(0.25 * 0.8 + 0.25 * 0.4 + 0.5 * 0.5)
    assert composite(1, 1, 1, w) == pytest.approx(1.0)
    assert composite(0, 0, 0, w) == 0.0


@pytest.mark.parametrize(
    ("reply", "expected"),
    [('{"score": 0.75, "reason": "x"}', 0.75), ('{"score": 7}', 1.0), ('{"score": -1}', 0.0),
     ('{"score": "high"}', 0.0), ('{"score": true}', 0.0), ("nope", 0.0)],
)  # fmt: skip
def test_completeness_parsing(reply, expected):
    assert ConfidenceScorer(FakeLLM(default=reply)).completeness("q", "a") == expected


def test_completeness_judge_failure_is_zero():
    class Broken:
        def complete(self, system, user):
            raise RuntimeError

    assert ConfidenceScorer(Broken()).completeness("q", "a") == 0.0


def test_scorer_combines_dimensions(chunks):
    answer = make_answer("Port is 7420 [1]. Uncited claim.", chunks)
    answer = answer.model_copy(
        update={"citations": [c.model_copy(update={"verified": True}) for c in answer.citations]}
    )
    judge = FakeLLM({"ANSWER TO RATE": '{"score": 1.0}'})
    conf = ConfidenceScorer(judge, ConfidenceWeights(retrieval=0.4, citation_coverage=0.4, completeness=0.2)).score(answer)
    assert (conf.retrieval, conf.citation_coverage, conf.completeness) == pytest.approx((0.8, 0.5, 1.0))
    assert conf.composite == pytest.approx(0.4 * 0.8 + 0.4 * 0.5 + 0.2 * 1.0)
    assert 0 <= conf.composite <= 1
