"""The composition root: the only place that imports from every module.

build_service() turns Settings into a working pipeline:

    upload -> loader -> save raw + text -> chunker -> indexer (Chroma + BM25)
    question -> retriever (hybrid or dense) -> answer flow (abstain | generate -> verify -> score)

Every component can be swapped for a fake, which is how the API tests run offline.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any

from rag.core.config import Settings
from rag.core.fakes import InMemoryVectorStore
from rag.core.interfaces import Chunker, Embedder, LLMClient, Reranker, Retriever
from rag.core.models import Answer, ChunkStrategy, DocumentSummary, IngestResult, RetrievalMode
from rag.generation.answering import AnswerFlow, build_answer_flow
from rag.generation.llm_client import make_llm_client
from rag.indexing.embedder import make_embedder
from rag.indexing.indexer import Indexer, build_indexer
from rag.indexing.sparse_index import BM25Index
from rag.ingestion.chunkers import make_chunker
from rag.ingestion.loaders import detect_format, load_document, save_document
from rag.retrieval.retriever import build_retriever

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class RagService:
    def __init__(self, settings: Settings, chunker: Chunker, indexer: Indexer, retriever: Retriever, flow: AnswerFlow) -> None:
        self.settings = settings
        self.chunker = chunker
        self.indexer = indexer
        self.retriever = retriever
        self.flow = flow
        self._write_lock = threading.Lock()  # the indexes are not safe for concurrent writes

    @property
    def strategy(self) -> ChunkStrategy:
        return self.settings.chunk_strategy

    # ----- questions -----

    def ask(self, question: str, mode: RetrievalMode | str = RetrievalMode.HYBRID, top_k: int | None = None) -> Answer:
        """Retrieve, then answer (or abstain). Also satisfies the evaluation Pipeline protocol."""
        chunks = self.retriever.retrieve(question, RetrievalMode(mode), top_k or self.settings.rerank_top_n)
        return self.flow.answer(question, chunks)

    # ----- documents -----

    def ingest_path(self, path: str | Path, root: str | Path | None = None) -> IngestResult:
        doc = save_document(load_document(path, root=root), self.settings.documents_path)
        chunks = self.chunker.chunk(doc)
        with self._write_lock:
            return self.indexer.index_document(doc.doc_id, chunks)

    def ingest_upload(self, filename: str | None, data: bytes) -> IngestResult:
        """Ingest uploaded bytes. The doc_id is the file's stem; any client path is dropped."""
        name = Path(filename or "").name
        detect_format(name)  # UnsupportedFormatError before touching the disk
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            return self.ingest_path(path, root=tmp)

    def list_documents(self) -> list[DocumentSummary]:
        return self.indexer.vector_store.list_docs()

    def health(self) -> dict[str, Any]:
        in_sync = self.indexer.in_sync()
        return {
            "status": "ok" if in_sync else "degraded",
            "strategy": self.strategy.value,
            "chunks": self.indexer.vector_store.count(),
            "indexes_in_sync": in_sync,
        }


def for_strategy(settings: Settings, strategy: ChunkStrategy | str | None) -> Settings:
    return settings if strategy is None else settings.model_copy(update={"chunk_strategy": ChunkStrategy(strategy)})


def build_ingestion(settings: Settings, embedder: Embedder, in_memory: bool = False) -> tuple[Chunker, Indexer]:
    """Chunker + indexer for settings.chunk_strategy. Needs no LLM (used by seed.py too)."""
    chunker = make_chunker(settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap, embedder)
    if in_memory:
        indexer = Indexer(embedder, InMemoryVectorStore(), BM25Index(), settings.dedup_threshold)
    else:
        indexer = build_indexer(settings, embedder)
    return chunker, indexer


def build_service(
    settings: Settings,
    strategy: ChunkStrategy | str | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    reranker: Reranker | None = None,
    in_memory: bool = False,
) -> RagService:
    """Real components by default; pass fakes (and in_memory=True) for offline tests."""
    settings = for_strategy(settings, strategy)
    embedder = embedder or make_embedder(settings)
    llm = llm or make_llm_client(settings)
    chunker, indexer = build_ingestion(settings, embedder, in_memory=in_memory)
    retriever = build_retriever(settings, embedder, indexer.vector_store, indexer.sparse_index, reranker)
    return RagService(settings, chunker, indexer, retriever, build_answer_flow(settings, llm, judge))
