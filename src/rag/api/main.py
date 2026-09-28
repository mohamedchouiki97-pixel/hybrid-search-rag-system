"""FastAPI app.

    uv run uvicorn rag.api.main:app --reload

The real service is built on startup from .env / environment settings. Tests pass
their own service to create_app().
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.routes import router
from rag.api.service import RagService, build_service
from rag.core.config import get_settings

DESCRIPTION = """Answers questions using only the indexed documents.

Every answer carries `[n]` citations that a judge model has checked, a confidence
breakdown, and a structured "I don't know" when retrieval finds nothing relevant."""


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            app.state.service = build_service(get_settings())
        yield

    app = FastAPI(title="Hybrid Search RAG", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
    app.state.service = service
    app.include_router(router)
    return app


app = create_app()
