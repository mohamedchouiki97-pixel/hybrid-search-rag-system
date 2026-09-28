"""Sparse search: BM25 keyword matching."""

from __future__ import annotations

from rag.core.interfaces import SparseIndex
from rag.core.models import RetrievedChunk


class SparseSearch:
    def __init__(self, index: SparseIndex) -> None:
        self.index = index

    def search(self, question: str, k: int) -> list[RetrievedChunk]:
        """Top k chunks, best first, with score = BM25 score and sparse_rank = 1..k."""
        if k <= 0:
            return []
        return [
            RetrievedChunk(chunk=chunk, score=score, sparse_rank=rank)
            for rank, (chunk, score) in enumerate(self.index.query(question, k), start=1)
        ]
