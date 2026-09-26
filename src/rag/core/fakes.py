"""Deterministic, offline stand-ins for the paid or heavy components.

Used by unit and integration tests so any module can be tested without the
others and without network calls.

Frozen after Step 0: request changes in CONTRACT_REQUESTS.md.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from collections.abc import Mapping, Sequence

from rag.core.interfaces import Vector
from rag.core.models import Chunk, DocumentSummary, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by the keyword fakes."""
    return _TOKEN_RE.findall(text.lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} vs {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


class FakeEmbedder:
    """Unit vectors seeded from a SHA-256 of the text.

    Same text always gives the same vector, in any process. Similar texts are not
    close. Pass `overrides` to hand-set vectors for specific texts (e.g. to plant a
    topic change for the semantic chunker).
    """

    def __init__(self, dim: int = 64, overrides: Mapping[str, Vector] | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.overrides: dict[str, Vector] = {}
        for text, vec in (overrides or {}).items():
            if len(vec) != dim:
                raise ValueError(f"override for {text!r} has dim {len(vec)}, expected {dim}")
            self.overrides[text] = list(vec)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> Vector:
        if text in self.overrides:
            return list(self.overrides[text])
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class FakeLLM:
    """Returns canned replies chosen by substring match on the prompt.

    `responses` maps a substring to a reply; the first key (in insertion order)
    found in the system or user prompt wins, otherwise `default` is returned.
    Every call is recorded in `calls` as (system, user). Thread safe.
    """

    def __init__(self, responses: Mapping[str, str] | None = None, default: str = "I don't know.") -> None:
        self.responses: dict[str, str] = dict(responses or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        with self._lock:
            self.calls.append((system, user))
        prompt = f"{system}\n{user}"
        for key, reply in self.responses.items():
            if key in prompt:
                return reply
        return self.default

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeReranker:
    """Scores by the number of distinct question words that appear in the chunk."""

    def rerank(self, question: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        q_tokens = set(tokenize(question))
        scored = []
        for rc in chunks:
            overlap = float(len(q_tokens & set(tokenize(rc.chunk.text))))
            scored.append(rc.model_copy(update={"rerank_score": overlap, "score": overlap}))
        scored.sort(key=lambda rc: rc.score, reverse=True)  # stable: ties keep input order
        return scored[:top_n]


class InMemoryVectorStore:
    """Brute-force cosine search over a dict. Upserts by chunk_id."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, Vector]] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._items[chunk.chunk_id] = (chunk, list(vec))

    def query(self, vector: Vector, k: int) -> list[tuple[Chunk, float]]:
        scored = [(chunk, cosine(vector, vec)) for chunk, vec in self._items.values()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._items)

    def ids(self) -> set[str]:
        return set(self._items)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._items.pop(cid, None)

    def delete_doc(self, doc_id: str) -> None:
        self.delete([cid for cid, (chunk, _) in self._items.items() if chunk.doc_id == doc_id])

    def list_docs(self) -> list[DocumentSummary]:
        summaries: dict[str, DocumentSummary] = {}
        for chunk, _ in self._items.values():
            s = summaries.get(chunk.doc_id)
            if s is None:
                summaries[chunk.doc_id] = DocumentSummary(
                    doc_id=chunk.doc_id, source_path=chunk.source_path, chunk_count=1
                )
            else:
                s.chunk_count += 1
        return sorted(summaries.values(), key=lambda s: s.doc_id)


class InMemorySparseIndex:
    """Keyword index scoring by query-term frequency in the chunk. Not BM25, but
    finds exact rare tokens the same way. Only chunks with a score above 0 are returned."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[str]]] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._items[chunk.chunk_id] = (chunk, tokenize(chunk.text))

    def query(self, text: str, k: int) -> list[tuple[Chunk, float]]:
        q_tokens = set(tokenize(text))
        scored = []
        for chunk, tokens in self._items.values():
            score = float(sum(1 for t in tokens if t in q_tokens))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def ids(self) -> set[str]:
        return set(self._items)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._items.pop(cid, None)

    def delete_doc(self, doc_id: str) -> None:
        self.delete([cid for cid, (chunk, _) in self._items.items() if chunk.doc_id == doc_id])
