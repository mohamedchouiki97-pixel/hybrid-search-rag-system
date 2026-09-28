import math

import pytest

from rag.core.interfaces import Reranker
from rag.core.models import RetrievedChunk
from rag.retrieval.reranker import CrossEncoderReranker, sigmoid


class StubCrossEncoder:
    """Returns a fixed logit per chunk text; records the pairs it was given."""

    def __init__(self, logits: dict[str, float]):
        self.logits = logits
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append(pairs)
        return [self.logits[text] for _, text in pairs]


@pytest.fixture
def chunks(make_chunk):
    return [RetrievedChunk(chunk=make_chunk(t, chunk_index=i), score=0.1) for i, t in enumerate(["low", "high", "mid"])]


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(2) == pytest.approx(1 / (1 + math.exp(-2)))
    assert sigmoid(-1000) == 0.0 and sigmoid(1000) == 1.0  # no overflow


def test_rerank_sorts_by_probability_and_truncates(chunks):
    model = StubCrossEncoder({"low": -3.0, "high": 4.0, "mid": 0.0})
    reranker = CrossEncoderReranker("any", model=model)
    out = reranker.rerank("q", chunks, top_n=2)
    assert [rc.chunk.text for rc in out] == ["high", "mid"]
    assert [rc.rerank_score for rc in out] == pytest.approx([sigmoid(4.0), 0.5])
    assert all(rc.score == rc.rerank_score and 0 <= rc.score <= 1 for rc in out)
    assert model.calls == [[("q", "low"), ("q", "high"), ("q", "mid")]]


def test_rerank_keeps_other_fields_and_input_untouched(chunks):
    chunks[1] = chunks[1].model_copy(update={"dense_rank": 2, "sparse_rank": 1})
    out = CrossEncoderReranker("any", model=StubCrossEncoder({"low": 0, "high": 1, "mid": 0})).rerank("q", chunks, 1)
    assert (out[0].dense_rank, out[0].sparse_rank) == (2, 1)
    assert chunks[1].rerank_score is None


def test_rerank_empty_and_zero_top_n(chunks):
    model = StubCrossEncoder({})
    reranker = CrossEncoderReranker("any", model=model)
    assert reranker.rerank("q", [], 5) == []
    assert reranker.rerank("q", chunks, 0) == []
    assert model.calls == []


def test_satisfies_protocol():
    assert isinstance(CrossEncoderReranker("any", model=StubCrossEncoder({})), Reranker)
