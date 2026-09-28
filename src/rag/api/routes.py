"""HTTP routes. Thin: validate input, call the service, map errors to status codes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from rag.api.service import MAX_UPLOAD_BYTES, RagService
from rag.core.models import Answer, AskRequest, DocumentSummary, IngestResult
from rag.ingestion.loaders import IngestionError, UnsupportedFormatError

router = APIRouter()


def get_service(request: Request) -> RagService:
    return request.app.state.service


Service = Annotated[RagService, Depends(get_service)]


class HealthResponse(BaseModel):
    status: str
    strategy: str
    chunks: int
    indexes_in_sync: bool


class ErrorResponse(BaseModel):
    detail: str


ASK_EXAMPLES = {
    "hybrid": {
        "summary": "Hybrid retrieval (default)",
        "value": {"question": "How do I declare an optional query parameter?", "mode": "hybrid", "top_k": 5},
    },
    "dense": {
        "summary": "Dense-only retrieval (for comparison)",
        "value": {"question": "How do I declare an optional query parameter?", "mode": "dense", "top_k": 5},
    },
}

ANSWER_EXAMPLE = {
    "question": "How do I declare an optional query parameter?",
    "answer_text": "Give the parameter a default value of None, e.g. q: str | None = None [1].",
    "citations": [
        {"marker": 1, "chunk_id": "3f2a9c...", "claim_text": "Give the parameter a default value of None, e.g. q: str | None = None.",
         "verified": True, "judge_reason": "The passage shows exactly this."}
    ],
    "confidence": {"retrieval": 0.94, "citation_coverage": 1.0, "completeness": 1.0, "composite": 0.98},
    "abstained": False,
    "found": "",
    "missing": "",
    "suggested_docs": [],
    "retrieved": [],
}  # fmt: skip


@router.get("/healthz", response_model=HealthResponse, tags=["system"], summary="Liveness and index consistency")
def healthz(service: Service) -> dict:
    return service.health()


@router.post(
    "/v1/ask",
    response_model=Answer,
    tags=["questions"],
    summary="Answer a question from the indexed documents, with verified citations",
    responses={200: {"content": {"application/json": {"example": ANSWER_EXAMPLE}}}},
)
def ask(request: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)], service: Service) -> Answer:
    return service.ask(request.question, request.mode, request.top_k)


@router.get("/v1/documents", response_model=list[DocumentSummary], tags=["documents"], summary="Indexed documents")
def documents(service: Service) -> list[DocumentSummary]:
    return service.list_documents()


@router.post(
    "/v1/ingest",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload a .md, .txt, .html or .pdf file and index it",
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Empty or unreadable file"},
    },
)
def ingest(file: UploadFile, service: Service) -> IngestResult:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"file exceeds {MAX_UPLOAD_BYTES // 2**20} MiB")
    try:
        return service.ingest_upload(file.filename, data)
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
