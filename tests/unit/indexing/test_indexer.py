import pytest

from rag.core.fakes import FakeEmbedder, InMemorySparseIndex, InMemoryVectorStore
from rag.core.models import ChunkStrategy
from rag.indexing.indexer import Indexer, IndexingError, build_indexer
from rag.indexing.sparse_index import BM25Index
from rag.indexing.vector_store import ChromaVectorStore

# Hand-set 3-d vectors: cosine to BASE is 1.0, ~0.990 and ~0.902.
BASE = [1.0, 0.0, 0.0]
NEAR = [0.99, 0.141, 0.0]
FAR = [0.9, 0.436, 0.0]
OTHER = [0.0, 0.0, 1.0]


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=3, overrides={"base": BASE, "near": NEAR, "far": FAR, "other": OTHER})


@pytest.fixture
def indexer(embedder):
    return Indexer(embedder, InMemoryVectorStore(), InMemorySparseIndex(), dedup_threshold=0.95)


def test_dedup_skips_above_threshold_keeps_below(indexer, make_chunk):
    chunks = [make_chunk(t, chunk_index=i) for i, t in enumerate(["base", "near", "far"])]
    result = indexer.index_document("doc1", chunks)
    assert (result.chunks_added, result.chunks_skipped_duplicate) == (2, 1)
    assert indexer.vector_store.ids() == {chunks[0].chunk_id, chunks[2].chunk_id}


def test_dedup_against_other_documents(indexer, make_chunk):
    indexer.index_document("doc1", [make_chunk("base", doc_id="doc1")])
    result = indexer.index_document("doc2", [make_chunk("near", doc_id="doc2"), make_chunk("other", doc_id="doc2", chunk_index=1)])
    assert (result.chunks_added, result.chunks_skipped_duplicate) == (1, 1)


def test_reindexing_a_document_replaces_it(indexer, make_chunk):
    v1 = [make_chunk(t, doc_id="doc1", chunk_index=i) for i, t in enumerate(["base", "other", "far"])]
    indexer.index_document("doc1", v1)
    v2 = [make_chunk("base", doc_id="doc1")]
    result = indexer.index_document("doc1", v2)
    assert (result.chunks_added, result.chunks_skipped_duplicate) == (1, 0)  # not a duplicate of itself
    assert indexer.vector_store.ids() == indexer.sparse_index.ids() == {v2[0].chunk_id}


def test_index_groups_by_document(indexer, make_chunk):
    chunks = [make_chunk("other", doc_id="b"), make_chunk("base", doc_id="a")]
    results = indexer.index(chunks)
    assert [(r.doc_id, r.chunks_added) for r in results] == [("a", 1), ("b", 1)]
    assert indexer.in_sync()


def test_rejects_chunks_from_another_document(indexer, make_chunk):
    with pytest.raises(ValueError, match="other-doc"):
        indexer.index_document("doc1", [make_chunk("x", doc_id="other-doc")])


def test_empty_chunk_list_clears_document(indexer, make_chunk):
    indexer.index_document("doc1", [make_chunk("base", doc_id="doc1")])
    result = indexer.index_document("doc1", [])
    assert result.chunks_added == 0 and indexer.vector_store.count() == 0


class FailingSparse(InMemorySparseIndex):
    def add(self, chunks):
        raise RuntimeError("sparse down")


class FailingVector(InMemoryVectorStore):
    def add(self, chunks, vectors):
        super().add(chunks[:1], vectors[:1])  # a partial write lands, then it fails
        raise RuntimeError("vector down")


def test_sparse_failure_rolls_back_vector_store(embedder, make_chunk):
    indexer = Indexer(embedder, InMemoryVectorStore(), FailingSparse())
    with pytest.raises(IndexingError, match="sparse index write failed"):
        indexer.index_document("doc1", [make_chunk("base", doc_id="doc1"), make_chunk("far", doc_id="doc1", chunk_index=1)])
    assert indexer.vector_store.ids() == indexer.sparse_index.ids() == set()


def test_vector_failure_rolls_back_partial_write(embedder, make_chunk):
    indexer = Indexer(embedder, FailingVector(), InMemorySparseIndex())
    with pytest.raises(IndexingError, match="vector store write failed"):
        indexer.index_document("doc1", [make_chunk("base", doc_id="doc1"), make_chunk("far", doc_id="doc1", chunk_index=1)])
    assert indexer.vector_store.ids() == indexer.sparse_index.ids() == set()
    assert indexer.in_sync()


def test_failure_keeps_other_documents(embedder, make_chunk):
    vector, sparse = InMemoryVectorStore(), InMemorySparseIndex()
    Indexer(embedder, vector, sparse).index_document("keep", [make_chunk("other", doc_id="keep")])
    with pytest.raises(IndexingError):
        Indexer(embedder, vector, FailingSparseAfter(sparse)).index_document("doc1", [make_chunk("base", doc_id="doc1")])
    assert vector.ids() == sparse.ids() == {make_chunk("other", doc_id="keep").chunk_id}


class FailingSparseAfter:
    """Wraps a working sparse index but fails on add."""

    def __init__(self, inner):
        self.inner = inner

    def add(self, chunks):
        raise RuntimeError("sparse down")

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_delete_doc_removes_from_both(indexer, make_chunk):
    indexer.index([make_chunk("base", doc_id="a"), make_chunk("other", doc_id="b")])
    indexer.delete_doc("a")
    assert {c.doc_id for c, _ in indexer.vector_store.query(OTHER, 5)} == {"b"}
    assert indexer.vector_store.ids() == indexer.sparse_index.ids()
    assert len(indexer.sparse_index.ids()) == 1


def test_build_indexer_uses_real_stores_per_strategy(settings):
    s = settings.model_copy(update={"chunk_strategy": ChunkStrategy.SEMANTIC})
    indexer = build_indexer(s, FakeEmbedder())
    assert isinstance(indexer.vector_store, ChromaVectorStore)
    assert isinstance(indexer.sparse_index, BM25Index)
    assert indexer.sparse_index.path.name == "semantic.json"
    assert indexer.dedup_threshold == s.dedup_threshold
