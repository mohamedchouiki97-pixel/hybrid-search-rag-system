import pytest
from pydantic import ValidationError

from rag.core.models import (
    Answer,
    AskRequest,
    Chunk,
    ChunkStrategy,
    Citation,
    Confidence,
    IngestResult,
    RetrievalMode,
    RetrievedChunk,
    make_chunk_id,
    stable_hash,
)


def test_stable_hash_is_deterministic_and_unambiguous():
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a|b", "c") != stable_hash("a", "b|c")
    assert len(stable_hash("x")) == 32
    assert len(stable_hash("x", length=12)) == 12


def test_make_chunk_id_is_stable_and_depends_on_every_part():
    base = make_chunk_id("doc1", ChunkStrategy.FIXED, 0)
    assert base == make_chunk_id("doc1", "fixed", 0)
    assert base != make_chunk_id("doc2", ChunkStrategy.FIXED, 0)
    assert base != make_chunk_id("doc1", ChunkStrategy.RECURSIVE, 0)
    assert base != make_chunk_id("doc1", ChunkStrategy.FIXED, 1)


def test_make_chunk_id_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        make_chunk_id("doc1", "bogus", 0)


def test_chunk_fills_char_count(make_chunk):
    assert make_chunk("hello").char_count == 5


def test_chunk_rejects_wrong_char_count(make_chunk):
    with pytest.raises(ValidationError, match="char_count"):
        make_chunk("hello", char_count=3)


def test_chunk_rejects_empty_text(make_chunk):
    with pytest.raises(ValidationError):
        make_chunk("")


def test_chunk_json_round_trip(make_chunk):
    chunk = make_chunk("text", section_heading="Intro", page_number=2)
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_ask_request_defaults():
    req = AskRequest(question="  what?  ")
    assert req.question == "what?"
    assert req.mode is RetrievalMode.HYBRID
    assert req.top_k == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "q", "top_k": 0},
        {"question": "q", "mode": "sparse"},
        {},
    ],
)
def test_ask_request_rejects_bad_input(payload):
    with pytest.raises(ValidationError):
        AskRequest(**payload)


@pytest.mark.parametrize("field", ["retrieval", "citation_coverage", "completeness", "composite"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_bounds(field, value):
    with pytest.raises(ValidationError):
        Confidence(**{field: value})


def test_citation_allows_unmatched_marker():
    c = Citation(marker=9, chunk_id=None, claim_text="claim")
    assert c.verified is None and c.judge_reason == ""


def test_answer_defaults_and_round_trip(make_chunk):
    rc = RetrievedChunk(chunk=make_chunk("t"), score=0.5, dense_rank=1)
    answer = Answer(
        question="q",
        answer_text="a [1]",
        citations=[Citation(marker=1, chunk_id=rc.chunk.chunk_id, claim_text="a")],
        retrieved=[rc],
    )
    assert answer.abstained is False
    assert answer.confidence == Confidence()
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


def test_ingest_result_rejects_negative_counts():
    with pytest.raises(ValidationError):
        IngestResult(doc_id="d", chunks_added=-1, chunks_skipped_duplicate=0)
