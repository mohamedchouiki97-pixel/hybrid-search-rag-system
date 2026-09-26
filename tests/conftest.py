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
def loader_fixtures_dir() -> Path:
    """Sample md/html/txt files exercising each loader's cleaning rules."""
    return FIXTURES_DIR / "loaders"


def build_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one text page per entry ("" makes a page with no text)."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for pid, text in zip(page_ids, pages, strict=True):
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line in text.split("\n") if text else []:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Write a PDF with the given page texts to tmp_path and return its path."""

    def _make(pages: list[str], name: str = "sample.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(build_pdf(pages))
        return path

    return _make


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
