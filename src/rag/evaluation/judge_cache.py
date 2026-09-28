"""Disk cache for judge calls, so re-running an evaluation is almost free.

Wraps any LLMClient. The key is hash(namespace, system, user); use the model name
as namespace so switching judge models never returns stale verdicts.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from rag.core.interfaces import LLMClient
from rag.core.models import stable_hash


class CachingLLMClient:
    def __init__(self, inner: LLMClient, cache_file: str | Path, namespace: str) -> None:
        self.inner = inner
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")
        self._db.commit()

    def complete(self, system: str, user: str) -> str:
        key = stable_hash(self.namespace, system, user)
        with self._lock:
            row = self._db.execute("SELECT reply FROM completions WHERE key = ?", (key,)).fetchone()
            if row is not None:
                self.hits += 1
                return row[0]
            self.misses += 1
        reply = self.inner.complete(system, user)  # outside the lock: calls run concurrently
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO completions (key, reply) VALUES (?, ?)", (key, reply))
            self._db.commit()
        return reply

    def close(self) -> None:
        with self._lock:
            self._db.close()
