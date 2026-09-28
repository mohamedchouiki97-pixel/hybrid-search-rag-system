import pytest
from pydantic import ValidationError

from rag.core.config import ConfidenceWeights, Settings
from rag.core.models import ChunkStrategy

DESIGN_KEYS = {
    "chunk_strategy", "chunk_size", "chunk_overlap", "dedup_threshold", "dense_k", "sparse_k",
    "rrf_dense_weight", "rrf_sparse_weight", "rerank_candidates", "rerank_top_n",
    "abstain_threshold", "confidence_weights", "llm_model", "embedding_model",
    "corpus_path", "chroma_path",
}  # fmt: skip


def make(**kwargs):
    return Settings(_env_file=None, **{"llm_model": "m", **kwargs})


def test_all_design_keys_exist():
    assert DESIGN_KEYS <= set(Settings.model_fields)


def test_defaults():
    s = make()
    assert s.chunk_strategy is ChunkStrategy.RECURSIVE
    assert s.dedup_threshold == 0.95
    assert (s.rrf_dense_weight, s.rrf_sparse_weight, s.rrf_k) == (0.7, 0.3, 60)
    assert (s.dense_k, s.rerank_candidates, s.rerank_top_n) == (10, 20, 5)
    assert s.embedding_model == "text-embedding-3-small"
    assert s.documents_path.as_posix() == "data/documents"


def test_llm_model_is_required(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None)


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.setenv("CHUNK_STRATEGY", "semantic")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CONFIDENCE_WEIGHTS", '{"retrieval": 1, "citation_coverage": 1, "completeness": 2}')
    s = Settings(_env_file=None)
    assert s.llm_model == "from-env"
    assert s.chunk_strategy is ChunkStrategy.SEMANTIC
    assert s.chunk_size == 500
    assert s.confidence_weights.normalized() == {"retrieval": 0.25, "citation_coverage": 0.25, "completeness": 0.5}


def test_env_file_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=from-file\nDENSE_K=7\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert (s.llm_model, s.dense_k) == ("from-file", 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_overlap": 800},
        {"rrf_dense_weight": 0, "rrf_sparse_weight": 0},
        {"rerank_top_n": 21},
        {"dedup_threshold": 1.5},
        {"chunk_strategy": "bogus"},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        make(**kwargs)


def test_default_confidence_weights_normalize_to_one():
    assert sum(ConfidenceWeights().normalized().values()) == pytest.approx(1.0)


def test_all_zero_confidence_weights_rejected():
    with pytest.raises(ValidationError):
        ConfidenceWeights(retrieval=0, citation_coverage=0, completeness=0)


def test_settings_fixture_is_isolated(settings, tmp_path):
    assert settings.llm_model == "fake-model"
    assert settings.chroma_path == tmp_path / "chroma"
    assert settings.documents_path == tmp_path / "documents"
