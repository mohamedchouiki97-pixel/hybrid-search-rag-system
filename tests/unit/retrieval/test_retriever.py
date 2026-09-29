import pytest

from rag.core.fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore
from rag.core.interfaces import Retriever
from rag.core.models import RetrievalMode
from rag.indexing.sparse_index import BM25Index
from rag.retrieval.dense import DenseSearch
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.retriever import HybridRetriever, build_retriever
from rag.retrieval.sparse import SparseSearch

QUESTION = "What does ERR-7Q42 mean?"
RARE = "Error ERR-7Q42 means a partition lease expired during a write."
OTHERS = [
    "The broker listens on port 7420 by default.",
    "Backups copy closed segment files to another location.",
    "Rolling upgrades start with the follower nodes.",
    "Messages are kept for 72 hours before deletion.",
    "Partitions are rebalanced after a node joins.",
]


def unit(i: int, dim: int = 8) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(dim)]


@pytest.fixture
def embedder():
    # Dense search is rigged to miss the rare-token chunk: the question points along
    # axis 0, the other chunks lean toward it, and the rare chunk is orthogonal.
    overrides = {QUESTION: unit(0), RARE: unit(7)}
    for i, text in enumerate(OTHERS, start=1):
        overrides[text] = [0.9, *unit(i)[1:]]
    return FakeEmbedder(dim=8, overrides=overrides)


@pytest.fixture
def corpus(make_chunk):
    return [make_chunk(t, doc_id=f"d{i}") for i, t in enumerate([RARE, *OTHERS])]


class SpyReranker(FakeReranker):
    def __init__(self):
        self.calls = []

    def rerank(self, question, chunks, top_n):
        self.calls.append(list(chunks))
        return super().rerank(question, chunks, top_n)


@pytest.fixture
def parts(embedder, corpus):
    store, sparse = InMemoryVectorStore(), BM25Index()
    store.add(corpus, embedder.embed([c.text for c in corpus]))
    sparse.add(corpus)
    return store, sparse


@pytest.fixture
def retriever(embedder, parts):
    store, sparse = parts
    return HybridRetriever(embedder, store, sparse, SpyReranker(), dense_k=3, sparse_k=3, rerank_candidates=4)


def test_dense_and_sparse_search_set_ranks(embedder, parts):
    store, sparse = parts
    dense = DenseSearch(embedder, store).search(QUESTION, 3)
    assert [r.dense_rank for r in dense] == [1, 2, 3] and all(r.sparse_rank is None for r in dense)
    hits = SparseSearch(sparse).search(QUESTION, 3)
    assert hits[0].chunk.text == RARE and hits[0].sparse_rank == 1
    assert DenseSearch(embedder, store).search(QUESTION, 0) == []
    assert SparseSearch(sparse).search(QUESTION, 0) == []


def test_bm25_finds_rare_token_that_dense_misses(retriever):
    dense = retriever.retrieve(QUESTION, RetrievalMode.DENSE, k=3)
    assert RARE not in [r.chunk.text for r in dense]
    hybrid = retriever.retrieve(QUESTION, RetrievalMode.HYBRID, k=3)
    assert hybrid[0].chunk.text == RARE
    assert hybrid[0].sparse_rank == 1 and hybrid[0].dense_rank is None


def test_hybrid_reranks_only_the_candidates(retriever):
    results = retriever.retrieve(QUESTION, RetrievalMode.HYBRID, k=2)
    [candidates] = retriever.reranker.calls
    assert len(candidates) == 4  # rerank_candidates
    assert len(results) == 2
    assert all(r.rerank_score is not None for r in results)


def test_dense_mode_keeps_dense_order_but_gets_reranker_scores(retriever, embedder, parts):
    results = retriever.retrieve(QUESTION, "dense", k=3)
    [scored] = retriever.reranker.calls
    assert len(scored) == 3  # scores only the k dense results; no fusion, no BM25
    assert [r.dense_rank for r in results] == [1, 2, 3]  # order untouched
    assert all(r.sparse_rank is None and r.rerank_score is not None for r in results)
    plain = DenseSearch(embedder, parts[0]).search(QUESTION, 3)
    assert [r.score for r in results] == [r.score for r in plain]  # score stays the cosine


def test_zero_k_and_empty_index(embedder):
    empty = HybridRetriever(embedder, InMemoryVectorStore(), BM25Index(), FakeReranker())
    assert empty.retrieve(QUESTION, RetrievalMode.HYBRID, 5) == []
    assert empty.retrieve(QUESTION, RetrievalMode.HYBRID, 0) == []


def test_build_retriever_from_settings(settings, embedder, parts):
    store, sparse = parts
    default = build_retriever(settings, embedder, store, sparse)
    assert isinstance(default, Retriever)
    assert isinstance(default.reranker, CrossEncoderReranker)
    assert default.reranker.model_name == settings.reranker_model
    assert default.reranker._model is None  # not downloaded until first use
    assert (default.dense_k, default.rerank_candidates, default.dense_weight) == (10, 20, 0.7)

    fake = FakeReranker()
    assert build_retriever(settings, embedder, store, sparse, reranker=fake).reranker is fake
