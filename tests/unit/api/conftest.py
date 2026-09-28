import pytest

from rag.api.service import build_service
from rag.core.fakes import FakeEmbedder, FakeLLM, FakeReranker
from rag.ingestion.loaders import iter_corpus_files

ANSWER = "The broker listens on port 7420 [1]."


@pytest.fixture
def fake_llm():
    return FakeLLM(
        {
            "CLAIM:": '{"supported": true, "reason": "stated"}',
            "ANSWER TO RATE": '{"score": 1, "reason": "complete"}',
            "Question:": ANSWER,
        }
    )


@pytest.fixture
def make_service(settings, fake_llm):
    """Service from fakes: FakeEmbedder, FakeLLM, FakeReranker, in-memory indexes."""

    def _make(strategy=None, corpus=None):
        service = build_service(
            settings, strategy, embedder=FakeEmbedder(), llm=fake_llm, reranker=FakeReranker(), in_memory=True
        )
        if corpus is not None:
            for path in iter_corpus_files(corpus):
                service.ingest_path(path, root=corpus)
        return service

    return _make


@pytest.fixture
def service(make_service, corpus_dir):
    return make_service(corpus=corpus_dir)
