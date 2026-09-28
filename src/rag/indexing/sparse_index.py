"""BM25 keyword index (rank_bm25), persisted as a JSON file of chunks.

rank_bm25 has no incremental add or delete: its statistics (document frequency,
average length) cover the whole corpus. So every change rebuilds the index from
the stored chunks. That is milliseconds at our size (a few thousand chunks).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.core.models import Chunk

_TOKEN_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens; snake_case identifiers also yield their parts.

    "response_model" -> ["response_model", "response", "model"], so a question
    about "response model" still matches code that says response_model.
    """
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p)
    return tokens


class BM25Index:
    """SparseIndex over BM25Okapi. With `path`, every change is saved atomically."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._chunks: dict[str, Chunk] = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._chunks = {c["chunk_id"]: Chunk.model_validate(c) for c in data}
        self._rebuild()

    # ----- SparseIndex -----

    def add(self, chunks: Sequence[Chunk]) -> None:
        updated = dict(self._chunks)
        updated.update({c.chunk_id: c for c in chunks})
        self._commit(updated)

    def query(self, text: str, k: int) -> list[tuple[Chunk, float]]:
        tokens = bm25_tokenize(text)
        if self._bm25 is None or not tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self._order)), key=lambda i: scores[i], reverse=True)
        return [(self._chunks[self._order[i]], float(scores[i])) for i in ranked[:k] if scores[i] > 0]

    def ids(self) -> set[str]:
        return set(self._chunks)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        drop = set(chunk_ids)
        if drop & self._chunks.keys():
            self._commit({cid: c for cid, c in self._chunks.items() if cid not in drop})

    def delete_doc(self, doc_id: str) -> None:
        self.delete([cid for cid, c in self._chunks.items() if c.doc_id == doc_id])

    # ----- internals -----

    def _commit(self, chunks: dict[str, Chunk]) -> None:
        """Save first, then swap in memory: a failed save leaves the index unchanged."""
        self._save(chunks)
        self._chunks = chunks
        self._rebuild()

    def _rebuild(self) -> None:
        self._order = list(self._chunks)
        corpus = [bm25_tokenize(self._chunks[cid].text) for cid in self._order]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _save(self, chunks: dict[str, Chunk]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = [c.model_dump(mode="json") for c in chunks.values()]
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: readers never see a half-written file
