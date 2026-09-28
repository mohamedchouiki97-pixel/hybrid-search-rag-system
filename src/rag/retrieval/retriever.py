"""Retriever: ties the search steps together.

hybrid: dense top dense_k + BM25 top sparse_k -> RRF -> top rerank_candidates
        -> cross-encoder -> top k
dense:  dense top k only (no fusion, no reranking), for the comparison toggle
"""

from __future__ import annotations

from rag.core.config import Settings
from rag.core.interfaces import Embedder, Reranker, SparseIndex, VectorStore
from rag.core.models import RetrievalMode, RetrievedChunk
from rag.retrieval.dense import DenseSearch
from rag.retrieval.fusion import rrf_fuse
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.sparse import SparseSearch


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        reranker: Reranker,
        dense_k: int = 10,
        sparse_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rerank_candidates: int = 20,
    ) -> None:
        self.dense = DenseSearch(embedder, vector_store)
        self.sparse = SparseSearch(sparse_index)
        self.reranker = reranker
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rerank_candidates = rerank_candidates

    def retrieve(self, question: str, mode: RetrievalMode, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        if RetrievalMode(mode) is RetrievalMode.DENSE:
            return self.dense.search(question, k)
        fused = rrf_fuse(
            self.dense.search(question, self.dense_k),
            self.sparse.search(question, self.sparse_k),
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            k=self.rrf_k,
        )
        return self.reranker.rerank(question, fused[: self.rerank_candidates], top_n=k)


def build_retriever(
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Retriever wired from config. Pass a reranker to override the real cross-encoder (e.g. in tests)."""
    return HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=sparse_index,
        reranker=reranker or CrossEncoderReranker(settings.reranker_model),
        dense_k=settings.dense_k,
        sparse_k=settings.sparse_k,
        rrf_k=settings.rrf_k,
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
        rerank_candidates=settings.rerank_candidates,
    )
