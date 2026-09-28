"""Pure functions behind the dashboard. They work on the API's JSON (plain dicts)."""

from __future__ import annotations

import re
from typing import Any

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

CONFIDENCE_LABELS = [
    ("retrieval", "Retrieval"),
    ("citation_coverage", "Citation coverage"),
    ("completeness", "Completeness"),
    ("composite", "Composite"),
]


def linkify_citations(text: str, anchor_prefix: str = "source") -> str:
    """Turn [1], [1][2] and [1, 2] into markdown links to #<prefix>-<n> anchors.

    A distinct prefix per answer keeps anchors unique when two answers are shown side by side.
    """

    def repl(m: re.Match[str]) -> str:
        return "".join(f"[[{n}]](#{anchor_prefix}-{n})" for n in (int(x) for x in m.group(1).split(",")))

    return _MARKER_RE.sub(repl, text)


def format_score(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def citation_status(verified: bool | None) -> str:
    return {True: "✅ supported", False: "❌ not supported"}.get(verified, "⏳ not checked")  # type: ignore[arg-type]


def confidence_rows(confidence: dict[str, float]) -> list[tuple[str, float]]:
    """(label, value in 0..1) for each dimension, composite last."""
    return [(label, min(1.0, max(0.0, float(confidence.get(key, 0.0))))) for key, label in CONFIDENCE_LABELS]


def source_title(chunk: dict[str, Any]) -> str:
    title = chunk["doc_id"]
    if chunk.get("section_heading"):
        title += f" — {chunk['section_heading']}"
    if chunk.get("page_number"):
        title += f" (p. {chunk['page_number']})"
    return title


def cited_markers(answer: dict[str, Any]) -> set[int]:
    return {c["marker"] for c in answer.get("citations", [])}


def chunk_table(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per retrieved chunk, ready for st.dataframe."""
    return [
        {
            "Rank": i,
            "Document": rc["chunk"]["doc_id"],
            "Section": rc["chunk"].get("section_heading") or "",
            "Score": format_score(rc.get("score"), 3),
            "Dense rank": rc.get("dense_rank") or "—",
            "BM25 rank": rc.get("sparse_rank") or "—",
            "Rerank": format_score(rc.get("rerank_score"), 3),
        }
        for i, rc in enumerate(retrieved, start=1)
    ]
