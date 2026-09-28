"""ChromaDB-backed VectorStore (persistent, file based, cosine similarity)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.core.interfaces import Vector
from rag.core.models import Chunk, DocumentSummary

_BATCH = 1000  # stay well under Chroma's max batch size


def _metadata(chunk: Chunk) -> dict[str, Any]:
    # Chroma metadata cannot hold None, so optional fields are simply left out.
    meta: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "strategy": chunk.strategy.value,
    }
    if chunk.section_heading is not None:
        meta["section_heading"] = chunk.section_heading
    if chunk.page_number is not None:
        meta["page_number"] = chunk.page_number
    return meta


def _chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=meta["doc_id"],
        text=text,
        source_path=meta["source_path"],
        section_heading=meta.get("section_heading"),
        page_number=meta.get("page_number"),
        chunk_index=meta["chunk_index"],
        strategy=meta["strategy"],
    )


class ChromaVectorStore:
    """One Chroma collection per chunking strategy (e.g. "chunks_recursive").

    Scores are cosine similarity (1 - Chroma's cosine distance), higher is better.
    Adding an existing chunk_id replaces it (upsert).
    """

    def __init__(self, path: str | Path, collection: str = "chunks") -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(
            name=collection,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we always pass our own vectors
        )

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[list(v) for v in vectors[i : i + _BATCH]],
                metadatas=[_metadata(c) for c in batch],
                documents=[c.text for c in batch],
            )

    def query(self, vector: Vector, k: int) -> list[tuple[Chunk, float]]:
        n = min(k, self.count())
        if n <= 0:
            return []
        res = self._col.query(
            query_embeddings=[list(vector)], n_results=n, include=["documents", "metadatas", "distances"]
        )
        return [
            (_chunk(cid, text, meta), 1.0 - dist)
            for cid, text, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def count(self) -> int:
        return self._col.count()

    def ids(self) -> set[str]:
        return set(self._col.get(include=[])["ids"])

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self._col.delete(ids=list(chunk_ids))

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def list_docs(self) -> list[DocumentSummary]:
        summaries: dict[str, DocumentSummary] = {}
        for meta in self._col.get(include=["metadatas"])["metadatas"] or []:
            s = summaries.get(meta["doc_id"])
            if s is None:
                summaries[meta["doc_id"]] = DocumentSummary(
                    doc_id=meta["doc_id"], source_path=meta["source_path"], chunk_count=1
                )
            else:
                s.chunk_count += 1
        return sorted(summaries.values(), key=lambda s: s.doc_id)
