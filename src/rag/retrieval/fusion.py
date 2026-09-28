"""Reciprocal Rank Fusion (RRF).

Dense scores (cosine) and BM25 scores live on different scales, so they cannot be
added. RRF uses only each chunk's rank in each list:

    score(chunk) = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)

A list the chunk is missing from contributes 0. The constant k (60 by convention)
flattens the curve so rank 1 does not dwarf rank 2, which is why a chunk found by
both searches usually beats one found by only one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.core.models import RetrievedChunk


def rrf_fuse(
    dense: Sequence[RetrievedChunk],
    sparse: Sequence[RetrievedChunk],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists (best first) into one, best first.

    Ranks are positions in the given lists (1-based). The output keeps each chunk's
    dense_rank and sparse_rank (None if absent) and sets score to its RRF score.
    Ties are broken by dense rank, then sparse rank, then chunk_id, so the order is
    deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("weights must be non-negative")

    fused: dict[str, dict[str, Any]] = {}

    def add(results: Sequence[RetrievedChunk], rank_field: str, weight: float) -> None:
        for rank, rc in enumerate(results, start=1):
            entry = fused.setdefault(
                rc.chunk.chunk_id, {"chunk": rc.chunk, "dense_rank": None, "sparse_rank": None, "score": 0.0}
            )
            if entry[rank_field] is None:  # a chunk listed twice counts once, at its best rank
                entry[rank_field] = rank
                entry["score"] += weight / (k + rank)

    add(dense, "dense_rank", dense_weight)
    add(sparse, "sparse_rank", sparse_weight)

    inf = float("inf")
    ordered = sorted(
        fused.values(),
        key=lambda e: (-e["score"], e["dense_rank"] or inf, e["sparse_rank"] or inf, e["chunk"].chunk_id),
    )
    return [RetrievedChunk(**e) for e in ordered]
