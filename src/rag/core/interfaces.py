"""Protocols every module codes against. Implementations never import each other directly.

Conventions:
- Every score is "higher is better".
- VectorStore scores are cosine similarity in [-1, 1].
- Stores upsert by chunk_id: adding an existing id replaces it.

Stable: edit only with all tests passing, and log the change in docs/CORE_CHANGELOG.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.core.models import (
    Answer,
    Chunk,
    Document,
    DocumentSummary,
    RetrievalMode,
    RetrievedChunk,
)

Vector = list[float]


@runtime_checkable
class Loader(Protocol):
    def load(self, path: str | Path) -> Document:
        """Read a file into a Document. Raises ValueError on an empty or unreadable file."""
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document. Chunk ids come from models.make_chunk_id."""
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """One vector per text, same order, same dimension."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def query(self, vector: Vector, k: int) -> list[tuple[Chunk, float]]:
        """Top k chunks by cosine similarity, best first."""
        ...

    def count(self) -> int: ...

    def ids(self) -> set[str]: ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove these ids. Unknown ids are ignored."""
        ...

    def delete_doc(self, doc_id: str) -> None: ...

    def list_docs(self) -> list[DocumentSummary]: ...


@runtime_checkable
class SparseIndex(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def query(self, text: str, k: int) -> list[tuple[Chunk, float]]:
        """Top k chunks by keyword score (BM25), best first."""
        ...

    def ids(self) -> set[str]: ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove these ids. Unknown ids are ignored."""
        ...

    def delete_doc(self, doc_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, question: str, mode: RetrievalMode, k: int) -> list[RetrievedChunk]:
        """Dense mode: dense only. Hybrid mode: dense + sparse, fused with RRF, then reranked."""
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, question: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Return the top_n best first, with rerank_score set and score = rerank_score."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Single-turn completion. Must be safe to call from multiple threads."""
        ...


@runtime_checkable
class Generator(Protocol):
    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        """Answer from chunks. Citations are parsed but verified is still None."""
        ...


@runtime_checkable
class CitationVerifier(Protocol):
    def verify(self, answer: Answer, chunks: Sequence[RetrievedChunk]) -> Answer:
        """Return a copy of answer with every citation's verified and judge_reason filled in."""
        ...
