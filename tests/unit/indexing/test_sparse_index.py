import pytest

from rag.core.interfaces import SparseIndex
from rag.indexing.sparse_index import BM25Index, bm25_tokenize

TEXTS = [
    "Error ERR-7Q42 means a partition lease expired",
    "Backups copy closed segment files",
    "Set response_model on the path operation",
    "The broker listens on port 7420",
    "Rolling upgrades start with follower nodes",
    "Messages are kept for 72 hours by default",
]


@pytest.fixture
def chunks(make_chunk):
    return [make_chunk(t, doc_id=f"d{i % 3}", chunk_index=i) for i, t in enumerate(TEXTS)]


def test_tokenize_splits_snake_case():
    assert bm25_tokenize("Use response_model, ERR-7Q42!") == ["use", "response_model", "response", "model", "err", "7q42"]


def test_protocol():
    assert isinstance(BM25Index(), SparseIndex)


def test_finds_exact_rare_token(chunks):
    idx = BM25Index()
    idx.add(chunks)
    results = idx.query("what does 7Q42 mean", k=3)
    assert results[0][0].chunk_id == chunks[0].chunk_id
    assert all(score > 0 for _, score in results)


def test_snake_case_matches_spaced_words(chunks):
    idx = BM25Index()
    idx.add(chunks)
    assert idx.query("response model", k=1)[0][0].chunk_id == chunks[2].chunk_id


def test_no_match_and_empty_cases(chunks):
    idx = BM25Index()
    assert idx.query("anything", k=5) == []  # empty index
    idx.add(chunks)
    assert idx.query("kafka", k=5) == []
    assert idx.query("!!!", k=5) == []
    assert idx.query("7q42", k=0) == []


def test_delete_and_delete_doc(chunks):
    idx = BM25Index()
    idx.add(chunks)
    idx.delete([chunks[0].chunk_id, "unknown"])
    assert chunks[0].chunk_id not in idx.ids()
    assert idx.query("7q42", k=5) == []
    idx.delete_doc("d1")
    assert idx.ids() == {c.chunk_id for c in chunks if c.doc_id != "d1"} - {chunks[0].chunk_id}


def test_add_upserts_by_id(chunks, make_chunk):
    idx = BM25Index()
    idx.add(chunks)
    idx.add([make_chunk("replaced text about kafka", doc_id="d0", chunk_index=0)])
    assert len(idx.ids()) == len(chunks)
    assert idx.query("kafka", k=1)[0][0].chunk_id == chunks[0].chunk_id


def test_persists_and_reloads(tmp_path, chunks):
    path = tmp_path / "bm25" / "recursive.json"
    BM25Index(path).add(chunks)
    reloaded = BM25Index(path)
    assert reloaded.ids() == {c.chunk_id for c in chunks}
    assert reloaded.query("7q42", k=1)[0][0] == chunks[0]
    assert not path.with_suffix(".json.tmp").exists()


def test_failed_save_leaves_index_unchanged(tmp_path, chunks, monkeypatch):
    idx = BM25Index(tmp_path / "bm25.json")
    idx.add(chunks[:3])

    def boom(_):
        raise OSError("disk full")

    monkeypatch.setattr(idx, "_save", boom)
    with pytest.raises(OSError):
        idx.add(chunks[3:])
    assert idx.ids() == {c.chunk_id for c in chunks[:3]}
