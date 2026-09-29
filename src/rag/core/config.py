"""Runtime configuration. Values come from init kwargs, then env vars, then .env, then defaults.

Stable: edit only with all tests passing, and log the change in CONTRACT_REQUESTS.md.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.core.models import ChunkStrategy


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ConfidenceWeights(BaseModel):
    retrieval: float = Field(default=0.4, ge=0.0)
    citation_coverage: float = Field(default=0.4, ge=0.0)
    completeness: float = Field(default=0.2, ge=0.0)

    @model_validator(mode="after")
    def _non_zero(self) -> ConfidenceWeights:
        if self.retrieval + self.citation_coverage + self.completeness <= 0:
            raise ValueError("confidence_weights must not all be zero")
        return self

    def normalized(self) -> dict[str, float]:
        """Weights rescaled to sum to 1, keyed by Confidence field name."""
        total = self.retrieval + self.citation_coverage + self.completeness
        return {
            "retrieval": self.retrieval / total,
            "citation_coverage": self.citation_coverage / total,
            "completeness": self.completeness / total,
        }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ingestion
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # Indexing
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # Retrieval
    dense_k: int = Field(default=10, gt=0)
    sparse_k: int = Field(default=10, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    rrf_dense_weight: float = Field(default=0.7, ge=0.0)
    rrf_sparse_weight: float = Field(default=0.3, ge=0.0)
    rerank_candidates: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)

    # Generation / confidence
    # Chosen from data: middle of the hybrid safe band 0.00-0.45 (reports/eval/threshold_sweep.md)
    abstain_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)

    # Models (llm_model has no default: it must be set in the environment)
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = Field(min_length=1)
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Paths
    corpus_path: Path = Path("corpus/fastapi_docs")
    chroma_path: Path = Path("data/chroma")
    documents_path: Path = Path("data/documents")  # raw + processed copy of every ingested file
    cache_path: Path = Path("data/cache")  # embedding cache (and later the judge cache)

    # Secrets (only needed by the real clients)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.rrf_dense_weight + self.rrf_sparse_weight <= 0:
            raise ValueError("rrf weights must not both be zero")
        if self.rerank_top_n > self.rerank_candidates:
            raise ValueError("rerank_top_n must not exceed rerank_candidates")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests should build Settings(...) directly instead."""
    return Settings()  # type: ignore[call-arg]
