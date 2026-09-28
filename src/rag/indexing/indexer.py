"""Indexer: embed, drop near-duplicates, write to both indexes, keep them in sync.

Write order is vector store, then sparse index. If the sparse write fails, the
chunks just written to the vector store are deleted again, so both indexes
always hold the same chunk ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

from rag.core.config import Settings
from rag.core.fakes import cosine
from rag.core.interfaces import Embedder, SparseIndex, Vector, VectorStore
from rag.core.models import Chunk, IngestResult
from rag.indexing.sparse_index import BM25Index
from rag.indexing.vector_store import ChromaVectorStore


class IndexingError(RuntimeError):
    """A write failed. Both indexes were rolled back to the same chunk ids."""


class Indexer:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        dedup_threshold: float = 0.95,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.dedup_threshold = dedup_threshold

    def index_document(self, doc_id: str, chunks: Sequence[Chunk]) -> IngestResult:
        """Replace everything indexed for doc_id with these chunks, skipping near-duplicates.

        Old chunks are deleted first, so an edited document never leaves stale
        chunks behind and is never flagged as a duplicate of its old version.
        """
        wrong = {c.doc_id for c in chunks} - {doc_id}
        if wrong:
            raise ValueError(f"chunks for {sorted(wrong)} passed to index_document({doc_id!r})")
        self.delete_doc(doc_id)
        if not chunks:
            return IngestResult(doc_id=doc_id, chunks_added=0, chunks_skipped_duplicate=0)

        vectors = self.embedder.embed([c.text for c in chunks])
        keep: list[Chunk] = []
        keep_vectors: list[Vector] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self._is_duplicate(vector, keep_vectors):
                continue
            keep.append(chunk)
            keep_vectors.append(vector)

        self._write(keep, keep_vectors)
        return IngestResult(doc_id=doc_id, chunks_added=len(keep), chunks_skipped_duplicate=len(chunks) - len(keep))

    def index(self, chunks: Sequence[Chunk]) -> list[IngestResult]:
        """Index chunks from any number of documents, one document at a time, in doc order."""
        ordered = sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index))
        return [self.index_document(doc_id, list(group)) for doc_id, group in groupby(ordered, key=lambda c: c.doc_id)]

    def delete_doc(self, doc_id: str) -> None:
        self.vector_store.delete_doc(doc_id)
        self.sparse_index.delete_doc(doc_id)

    def in_sync(self) -> bool:
        return self.vector_store.ids() == self.sparse_index.ids()

    # ----- internals -----

    def _is_duplicate(self, vector: Vector, batch_vectors: Sequence[Vector]) -> bool:
        """Near-duplicate of a chunk earlier in this batch, or of any chunk already indexed."""
        if any(cosine(vector, v) >= self.dedup_threshold for v in batch_vectors):
            return True
        nearest = self.vector_store.query(vector, 1)
        return bool(nearest) and nearest[0][1] >= self.dedup_threshold

    def _write(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        try:
            self.vector_store.add(chunks, vectors)
        except Exception as exc:
            self.vector_store.delete(ids)  # a partial batch may have landed
            raise IndexingError(f"vector store write failed; rolled back {len(ids)} chunks") from exc
        try:
            self.sparse_index.add(chunks)
        except Exception as exc:
            self.vector_store.delete(ids)
            self.sparse_index.delete(ids)
            raise IndexingError(f"sparse index write failed; rolled back {len(ids)} chunks") from exc


def build_indexer(settings: Settings, embedder: Embedder) -> Indexer:
    """Chroma + BM25 stores for the configured chunking strategy (one pair per strategy)."""
    strategy = settings.chunk_strategy.value
    return Indexer(
        embedder=embedder,
        vector_store=ChromaVectorStore(settings.chroma_path, collection=f"chunks_{strategy}"),
        sparse_index=BM25Index(settings.chroma_path.parent / "bm25" / f"{strategy}.json"),
        dedup_threshold=settings.dedup_threshold,
    )
