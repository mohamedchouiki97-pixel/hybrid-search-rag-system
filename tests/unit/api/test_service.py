from pathlib import Path

import pytest

from rag.api.service import build_ingestion, build_service, for_strategy
from rag.core.fakes import FakeEmbedder, FakeLLM, FakeReranker
from rag.core.models import ChunkStrategy, RetrievalMode
from rag.indexing.sparse_index import BM25Index
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.loaders import UnsupportedFormatError


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_service_builds_with_each_strategy(make_service, corpus_dir, strategy):
    service = make_service(strategy=strategy, corpus=corpus_dir)
    assert service.strategy is strategy
    assert service.health()["indexes_in_sync"] is True
    answer = service.ask("What port does the broker listen on?")
    assert answer.retrieved and all(rc.chunk.strategy is strategy for rc in answer.retrieved)


def test_ingest_saves_raw_and_text(service, settings):
    doc_dir = Path(settings.documents_path) / "overview"
    assert (doc_dir / "raw.md").exists() and (doc_dir / "text.md").exists()


def test_reingest_replaces_instead_of_duplicating(service, corpus_dir):
    before = service.health()["chunks"]
    result = service.ingest_path(corpus_dir / "errors.md", root=corpus_dir)
    assert result.chunks_added > 0 and result.chunks_skipped_duplicate == 0
    assert service.health()["chunks"] == before


def test_ingest_upload_rejects_unsupported_before_writing(service, settings):
    with pytest.raises(UnsupportedFormatError):
        service.ingest_upload("x.exe", b"MZ")
    assert not (Path(settings.documents_path) / "x").exists()


def test_top_k_defaults_to_rerank_top_n(service):
    answer = service.ask("What port does the broker listen on?", RetrievalMode.DENSE)
    assert len(answer.retrieved) == service.settings.rerank_top_n


def test_health_reports_degraded_when_out_of_sync(service, make_chunk):
    service.indexer.vector_store.add([make_chunk("stray")], [[0.0] * 64])
    assert service.health()["status"] == "degraded"


def test_real_stores_are_used_by_default(settings):
    chunker, indexer = build_ingestion(for_strategy(settings, "fixed"), FakeEmbedder())
    assert isinstance(indexer.vector_store, ChromaVectorStore) and isinstance(indexer.sparse_index, BM25Index)
    service = build_service(settings, embedder=FakeEmbedder(), llm=FakeLLM(), reranker=FakeReranker())
    assert isinstance(service.indexer.vector_store, ChromaVectorStore)


def test_for_strategy():
    class S:
        chunk_strategy = ChunkStrategy.FIXED

    s = S()
    assert for_strategy(s, None) is s  # type: ignore[arg-type]
