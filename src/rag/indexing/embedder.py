"""Embedders: the real OpenAI client plus a disk cache that works with any Embedder."""

from __future__ import annotations

import sqlite3
import threading
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.core.config import Settings
from rag.core.interfaces import Embedder, Vector
from rag.core.models import stable_hash


class OpenAIEmbedder:
    """OpenAI embeddings API, called in batches. The client is created on first use."""

    def __init__(self, model: str, api_key: str | None = None, batch_size: int = 256, client: Any = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.batch_size = batch_size
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(list(d.embedding) for d in sorted(response.data, key=lambda d: d.index))
        return vectors


class CachingEmbedder:
    """Wraps any Embedder with a SQLite cache keyed by hash(namespace, text).

    Re-indexing, re-running evals and semantic chunking then never pay twice for
    the same text. `namespace` should be the model name, so switching models never
    returns stale vectors. Vectors are stored as float32 (plenty for cosine).
    """

    def __init__(self, inner: Embedder, cache_file: str | Path, namespace: str) -> None:
        self.inner = inner
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self._db.commit()

    def _key(self, text: str) -> str:
        return stable_hash(self.namespace, text)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        keys = [self._key(t) for t in texts]
        found = self._lookup(set(keys))
        missing: dict[str, str] = {}  # key -> text, deduplicated, in first-seen order
        for key, text in zip(keys, texts, strict=True):
            if key not in found and key not in missing:
                missing[key] = text
        n_missing = sum(1 for k in keys if k not in found)
        self.hits += len(keys) - n_missing
        self.misses += n_missing
        if missing:
            fresh = self.inner.embed(list(missing.values()))
            # Round to float32 now, so a text gives the same vector whether or not it was cached.
            new = {k: array("f", v).tolist() for k, v in zip(missing, fresh, strict=True)}
            self._store(new)
            found.update(new)
        return [list(found[k]) for k in keys]

    def _lookup(self, keys: set[str]) -> dict[str, Vector]:
        if not keys:
            return {}
        found: dict[str, Vector] = {}
        key_list = list(keys)
        with self._lock:
            for i in range(0, len(key_list), 500):  # SQLite parameter limit
                chunk = key_list[i : i + 500]
                rows = self._db.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})", chunk
                ).fetchall()
                for key, blob in rows:
                    found[key] = array("f", blob).tolist()
        return found

    def _store(self, vectors: dict[str, Vector]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(k, array("f", v).tobytes()) for k, v in vectors.items()],
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def make_embedder(settings: Settings) -> CachingEmbedder:
    """The production embedder: OpenAI behind a disk cache in settings.cache_path."""
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    inner = OpenAIEmbedder(settings.embedding_model, api_key=api_key)
    return CachingEmbedder(inner, settings.cache_path / "embeddings.sqlite", namespace=settings.embedding_model)
