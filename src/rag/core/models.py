"""Shared data models. Every module exchanges data using only these types.

Stable: edit only with all tests passing, and log the change in CONTRACT_REQUESTS.md.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class DocFormat(StrEnum):
    MD = "md"
    TXT = "txt"
    HTML = "html"
    PDF = "pdf"


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    DENSE = "dense"


Score01 = Annotated[float, Field(ge=0.0, le=1.0)]


def stable_hash(*parts: object, length: int = 32) -> str:
    """Deterministic hex id from parts. Unambiguous: ("a|b", "c") != ("a", "b|c")."""
    payload = json.dumps([str(p) for p in parts], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_chunk_id(doc_id: str, strategy: ChunkStrategy | str, chunk_index: int) -> str:
    """Stable chunk id: re-ingesting the same doc with the same strategy gives the same ids."""
    return stable_hash(doc_id, ChunkStrategy(strategy).value, chunk_index)


class Document(BaseModel):
    doc_id: str
    source_path: str
    format: DocFormat
    raw_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str = Field(min_length=1)
    source_path: str
    section_heading: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    strategy: ChunkStrategy
    char_count: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _fill_char_count(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("char_count") is None and isinstance(data.get("text"), str):
            data = {**data, "char_count": len(data["text"])}
        return data

    @model_validator(mode="after")
    def _check_char_count(self) -> Chunk:
        if self.char_count != len(self.text):
            raise ValueError(f"char_count {self.char_count} does not match len(text) {len(self.text)}")
        return self


class RetrievedChunk(BaseModel):
    """A chunk plus how it was found. `score` is the current ranking score (higher is better)."""

    chunk: Chunk
    score: float
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None


class Citation(BaseModel):
    """A [n] marker in the answer. chunk_id is None when n has no matching retrieved chunk."""

    marker: int = Field(ge=0)  # 0 is never valid, but an LLM can write [0]; it is kept so it can be flagged
    chunk_id: str | None
    claim_text: str
    verified: bool | None = None
    judge_reason: str = ""


class Confidence(BaseModel):
    retrieval: Score01 = 0.0
    citation_coverage: Score01 = 0.0
    completeness: Score01 = 0.0
    composite: Score01 = 0.0


class Answer(BaseModel):
    question: str
    answer_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    abstained: bool = False
    found: str = ""
    missing: str = ""
    suggested_docs: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "What does error ERR-7Q42 mean?", "mode": "hybrid", "top_k": 5},
            ]
        }
    )

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=50)


class IngestResult(BaseModel):
    doc_id: str
    chunks_added: int = Field(ge=0)
    chunks_skipped_duplicate: int = Field(ge=0)


class DocumentSummary(BaseModel):
    """One indexed document, as listed by VectorStore.list_docs and GET /v1/documents."""

    doc_id: str
    source_path: str
    chunk_count: int = Field(ge=0)
