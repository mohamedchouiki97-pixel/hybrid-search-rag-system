"""Abstention: when retrieval is weak, say "I don't know" in a structured way instead of guessing.

Built from the retrieved chunks alone (no LLM call), so abstaining is free and can
never hallucinate.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.core.config import ConfidenceWeights
from rag.core.models import Answer, Confidence, RetrievedChunk
from rag.generation.confidence import composite

ABSTAIN_TEXT = "I don't know based on the provided documents."


def should_abstain(retrieval: float, threshold: float) -> bool:
    return retrieval < threshold


def build_abstain_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    retrieval: float,
    threshold: float,
    weights: ConfidenceWeights | None = None,
    max_suggestions: int = 3,
) -> Answer:
    suggested: list[str] = []
    places: list[str] = []
    for rc in chunks:
        c = rc.chunk
        if c.doc_id in suggested:
            continue
        suggested.append(c.doc_id)
        places.append(f"{c.section_heading} ({c.doc_id})" if c.section_heading else c.doc_id)
        if len(suggested) == max_suggestions:
            break

    found = f"Closest matches: {'; '.join(places)}." if places else "No related passages were found."
    missing = (
        f"No passage answers the question closely enough "
        f"(retrieval confidence {retrieval:.2f}, threshold {threshold:.2f})."
    )
    return Answer(
        question=question,
        answer_text=ABSTAIN_TEXT,
        confidence=Confidence(retrieval=retrieval, composite=composite(retrieval, 0.0, 0.0, weights or ConfidenceWeights())),
        abstained=True,
        found=found,
        missing=missing,
        suggested_docs=suggested,
        retrieved=list(chunks),
    )
