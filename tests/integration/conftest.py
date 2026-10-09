"""Integration fixtures: real ChromaDB (temp dir), real BM25, real chunkers and service;
FakeEmbedder and FakeLLM for the paid APIs. No network.

The real cross-encoder is used where scores must be realistic (abstention). It is
loaded offline from the local Hugging Face cache; tests that need it skip if it is
not cached yet (it is cached after the first `seed.py` or API run).
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # before huggingface_hub is imported anywhere

import pytest

from rag.api.service import build_service
from rag.core.fakes import FakeEmbedder, FakeLLM, FakeReranker
from rag.ingestion.loaders import iter_corpus_files
from rag.retrieval.reranker import CrossEncoderReranker

SUPPORTED = '{"supported": true, "reason": "stated in the passage"}'
COMPLETE = '{"score": 1, "reason": "complete"}'


def _model_is_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)


@pytest.fixture(scope="session")
def cross_encoder():
    """The real reranker, loaded once per session from the local cache."""
    model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if not _model_is_cached(model):
        pytest.skip(f"{model} is not in the local Hugging Face cache")
    reranker = CrossEncoderReranker(model)
    reranker.model  # load now, offline  # noqa: B018
    return reranker


@pytest.fixture
def llm():
    return FakeLLM({"CLAIM:": SUPPORTED, "ANSWER TO RATE": COMPLETE}, default="The broker listens on port 7420 [1].")


@pytest.fixture
def make_service(settings, llm):
    """Service with real Chroma + BM25 under tmp_path. Ingests the fixture corpus unless ingest=False."""

    def _make(strategy=None, reranker=None, embedder=None, ingest=True, corpus=None, abstain_threshold=None):
        s = (
            settings
            if abstain_threshold is None
            else settings.model_copy(update={"abstain_threshold": abstain_threshold})
        )
        service = build_service(
            s, strategy, embedder=embedder or FakeEmbedder(), llm=llm, reranker=reranker or FakeReranker()
        )
        if ingest:
            for path in iter_corpus_files(corpus):
                service.ingest_path(path, root=corpus)
        return service

    return _make
