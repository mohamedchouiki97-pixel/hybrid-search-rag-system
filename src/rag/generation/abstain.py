"""Abstention: say "I don't know" in a structured way instead of guessing.

Two layers trigger it:
1. Retrieval gate: retrieval confidence below the threshold, so the LLM is never called.
2. Model refusal: retrieval looked fine, but the model replied with the exact
   "documents do not answer" sentence the prompt asks for.

Either way the answer is built from the retrieved chunks alone, so it cannot hallucinate.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.core.config import ConfidenceWeights
from rag.core.models import Answer, Confidence, RetrievedChunk
from rag.generation.confidence import composite
from rag.generation.prompts import NO_ANSWER_SENTENCE

ABSTAIN_TEXT = "I don't know based on the provided documents."
REFUSAL_MISSING = "The retrieved passages are related, but none of them answers the question."


def should_abstain(retrieval: float, threshold: float) -> bool:
    return retrieval < threshold


def is_refusal(answer_text: str) -> bool:
    """True if the model gave the full "no answer" reply (not a partial answer that mentions gaps)."""
    normalized = answer_text.strip().strip("\"'`").strip().rstrip(".").casefold()
    return normalized == NO_ANSWER_SENTENCE.rstrip(".").casefold()


def build_abstain_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    retrieval: float,
    threshold: float,
    weights: ConfidenceWeights | None = None,
    max_suggestions: int = 3,
    missing: str | None = None,
) -> Answer:
    """missing defaults to the retrieval-gate explanation; pass REFUSAL_MISSING for a model refusal."""
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
    missing = missing or (
        f"No passage answers the question closely enough "
        f"(retrieval confidence {retrieval:.2f}, threshold {threshold:.2f})."
    )
    return Answer(
        question=question,
        answer_text=ABSTAIN_TEXT,
        confidence=Confidence(
            retrieval=retrieval, composite=composite(retrieval, 0.0, 0.0, weights or ConfidenceWeights())
        ),
        abstained=True,
        found=found,
        missing=missing,
        suggested_docs=suggested,
        retrieved=list(chunks),
    )
