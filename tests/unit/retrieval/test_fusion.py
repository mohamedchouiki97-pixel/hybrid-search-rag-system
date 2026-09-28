import pytest

from rag.core.models import RetrievedChunk
from rag.retrieval.fusion import rrf_fuse


@pytest.fixture
def rc(make_chunk):
    def _rc(name: str) -> RetrievedChunk:
        return RetrievedChunk(chunk=make_chunk(name, doc_id=name), score=0.0)

    return _rc


def names(results):
    return [r.chunk.text for r in results]


def test_rrf_math_on_hand_written_lists(rc):
    # k=60, weights 0.7/0.3:
    #   A = .7/61 = .011475   B = .7/62 = .011290
    #   C = .7/63 + .3/61 = .016029 (in both lists)   D = .3/62 = .004839
    fused = rrf_fuse([rc("A"), rc("B"), rc("C")], [rc("C"), rc("D")], 0.7, 0.3, k=60)
    assert names(fused) == ["C", "A", "B", "D"]
    assert [r.score for r in fused] == pytest.approx([0.7 / 63 + 0.3 / 61, 0.7 / 61, 0.7 / 62, 0.3 / 62])
    c = fused[0]
    assert (c.dense_rank, c.sparse_rank) == (3, 1)
    assert (fused[3].dense_rank, fused[3].sparse_rank) == (None, 2)


def test_weights_change_the_order(rc):
    dense, sparse = [rc("A"), rc("B"), rc("C")], [rc("C"), rc("D")]
    assert names(rrf_fuse(dense, sparse, 0.3, 0.7)) == ["C", "D", "A", "B"]
    assert names(rrf_fuse(dense, sparse, 1.0, 0.0))[:3] == ["A", "B", "C"]


def test_item_in_both_lists_beats_item_in_one(rc):
    dense = [rc(n) for n in "ABCDEFGHIJ"]
    sparse = [rc(n) for n in "KLMNOPQRSJ"]  # J is last in both lists
    fused = rrf_fuse(dense, sparse, 0.5, 0.5)
    assert names(fused)[0] == "J"


def test_ties_are_deterministic(rc):
    assert names(rrf_fuse([rc("A")], [rc("B")], 0.5, 0.5)) == ["A", "B"]  # dense rank breaks the tie


def test_duplicates_in_one_list_count_once(rc):
    fused = rrf_fuse([rc("A"), rc("A")], [], 1.0, 0.0)
    assert len(fused) == 1 and fused[0].score == pytest.approx(1 / 61)


def test_empty_inputs(rc):
    assert rrf_fuse([], []) == []
    assert names(rrf_fuse([], [rc("A")])) == ["A"]


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"dense_weight": -1}, {"sparse_weight": -0.1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rrf_fuse([], [], **kwargs)
