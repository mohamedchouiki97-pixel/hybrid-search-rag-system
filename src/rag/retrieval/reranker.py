"""Cross-encoder reranker.

Embeddings encode the question and a chunk separately, then compare vectors. A
cross-encoder reads (question, chunk) together, so it can judge relevance far
more precisely, but it is too slow to run over the whole corpus. So it only
rescores the top candidates from fusion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rag.core.models import RetrievedChunk


def sigmoid(x: float) -> float:
    # Numerically stable for large |x|.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded on first use (downloads ~90 MB once).

    The model is asked for raw logits and the sigmoid is applied here, exactly once,
    so rerank_score is always in 0..1 whatever activation the model's config names.
    """

    def __init__(self, model_name: str, batch_size: int = 32, model: Any = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, activation_fn=torch.nn.Identity())
        return self._model

    def rerank(self, question: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks or top_n <= 0:
            return []
        logits = self.model.predict(
            [(question, rc.chunk.text) for rc in chunks], batch_size=self.batch_size, show_progress_bar=False
        )
        rescored = []
        for rc, logit in zip(chunks, logits, strict=True):
            p = sigmoid(float(logit))
            rescored.append(rc.model_copy(update={"rerank_score": p, "score": p}))
        rescored.sort(key=lambda rc: rc.score, reverse=True)  # stable: ties keep fusion order
        return rescored[:top_n]
