"""Dense search: embed the question, nearest chunks by cosine similarity."""

from __future__ import annotations

from rag.core.interfaces import Embedder, VectorStore
from rag.core.models import RetrievedChunk


class DenseSearch:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, question: str, k: int) -> list[RetrievedChunk]:
        """Top k chunks, best first, with score = cosine and dense_rank = 1..k."""
        if k <= 0:
            return []
        (vector,) = self.embedder.embed([question])
        return [
            RetrievedChunk(chunk=chunk, score=score, dense_rank=rank)
            for rank, (chunk, score) in enumerate(self.store.query(vector, k), start=1)
        ]
