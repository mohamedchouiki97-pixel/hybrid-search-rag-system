"""Shared test setup.

- Tests are marked by folder: tests/unit -> unit, tests/integration -> integration,
  tests/live -> live. No need to add the marker by hand.
- Unit tests cannot open network connections (localhost is allowed, asyncio needs it).
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag.core.config import Settings
from rag.core.fakes import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    InMemorySparseIndex,
    InMemoryVectorStore,
)
from rag.core.models import Chunk, ChunkStrategy, make_chunk_id

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"

_LAYER_MARKERS = ("unit", "integration", "live")
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.path)).parts
        for layer in _LAYER_MARKERS:
            if layer in parts:
                item.add_marker(getattr(pytest.mark, layer))


# ---------- network guard ----------


class NetworkBlockedError(RuntimeError):
    pass


def _is_local(address: Any) -> bool:
    if isinstance(address, tuple) and address:
        return str(address[0]) in _LOCAL_HOSTS
    return isinstance(address, str | bytes)  # AF_UNIX paths


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("unit") is None:
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if not _is_local(address):
            raise NetworkBlockedError(f"unit tests may not open network connections (to {address!r})")
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if not _is_local(address):
            raise NetworkBlockedError(f"unit tests may not open network connections (to {address!r})")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and str(host) not in _LOCAL_HOSTS:
            raise NetworkBlockedError(f"unit tests may not resolve hosts ({host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


# ---------- corpus ----------


@pytest.fixture
def corpus_dir() -> Path:
    """The 5-doc fixture corpus (read only; copy to tmp_path before writing)."""
    return CORPUS_DIR


@pytest.fixture
def known_facts() -> dict[str, list[dict[str, Any]]]:
    """Facts planted in the fixture corpus, grouped by type (lookup, rare_token, multi_hop, no_answer)."""
    return json.loads((FIXTURES_DIR / "known_facts.json").read_text(encoding="utf-8"))


# ---------- config ----------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings isolated from the developer's .env, with data paths in a temp dir."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_model="fake-model",
        corpus_path=CORPUS_DIR,
        chroma_path=tmp_path / "chroma",
    )


# ---------- fakes ----------


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def sparse_index() -> InMemorySparseIndex:
    return InMemorySparseIndex()


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    """Factory for valid chunks: make_chunk("some text", doc_id="d1", chunk_index=0, ...)."""

    def _make(
        text: str,
        doc_id: str = "doc1",
        chunk_index: int = 0,
        strategy: ChunkStrategy = ChunkStrategy.FIXED,
        **overrides: Any,
    ) -> Chunk:
        fields: dict[str, Any] = {
            "chunk_id": make_chunk_id(doc_id, strategy, chunk_index),
            "doc_id": doc_id,
            "text": text,
            "source_path": f"{doc_id}.md",
            "chunk_index": chunk_index,
            "strategy": strategy,
        }
        fields.update(overrides)
        return Chunk(**fields)

    return _make
