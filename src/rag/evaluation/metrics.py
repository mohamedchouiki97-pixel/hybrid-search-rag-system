"""Evaluation metrics.

Retrieval (no LLM):
- recall@k:  share of gold sections that appear among the top k retrieved chunks
- MRR:       1 / rank of the first retrieved chunk from any gold section (0 if none)

A retrieved chunk matches a gold section when its doc_id matches and its
section_heading is that section or one of its subsections ("A > B" matches gold "A").

Answer quality (judge LLM):
- correctness:   answer vs gold answer, 0 / 0.5 / 1. For no_answer questions the
                 answer is correct only if the system abstained (no judge call).
- faithfulness:  share of the answer's claims supported by the retrieved context.

From the verifier:
- citation accuracy: share of citations the verifier marked supported.

A metric that does not apply to a question is None and is left out of averages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from rag.core.interfaces import LLMClient
from rag.core.models import Answer, RetrievedChunk
from rag.evaluation.golden import GoldenItem, GoldSection, QuestionType

# Evaluation grades any pipeline, so it keeps its own small helpers instead of
# importing generation's (modules only share core).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {rc.chunk.text}" for i, rc in enumerate(chunks, start=1))


# ---------- retrieval ----------


def matches(rc: RetrievedChunk, gold: GoldSection) -> bool:
    heading = rc.chunk.section_heading or ""
    return rc.chunk.doc_id == gold.doc_id and (heading == gold.section or heading.startswith(gold.section + " > "))


def recall_at_k(retrieved: Sequence[RetrievedChunk], gold: Sequence[GoldSection], k: int) -> float | None:
    if not gold:
        return None
    top = retrieved[:k]
    return sum(1 for g in gold if any(matches(rc, g) for rc in top)) / len(gold)


def mrr(retrieved: Sequence[RetrievedChunk], gold: Sequence[GoldSection]) -> float | None:
    if not gold:
        return None
    for rank, rc in enumerate(retrieved, start=1):
        if any(matches(rc, g) for g in gold):
            return 1.0 / rank
    return 0.0


# ---------- citations ----------


def citation_accuracy(answer: Answer) -> float | None:
    if not answer.citations:
        return None
    return sum(1 for c in answer.citations if c.verified) / len(answer.citations)


# ---------- judge-based ----------

CORRECTNESS_SYSTEM = """You grade a SYSTEM ANSWER against a GOLD ANSWER for a QUESTION about software documentation.
Score 1 if the system answer contains the key facts of the gold answer and nothing that contradicts it.
Score 0.5 if it is partly right: some key facts are missing, but nothing important is wrong.
Score 0 if it is wrong, contradicts the gold answer, or does not answer.
Extra detail that is consistent with the gold answer is fine.
Reply with JSON only: {"score": 0 or 0.5 or 1, "reason": "<one short sentence>"}"""

AMBIGUOUS_RULE = """
This question is AMBIGUOUS. Score 1 if the system answer either asks which meaning was intended, or covers
the main interpretations described in the gold answer. Score 0.5 if it covers only one interpretation
correctly without acknowledging the others."""

FAITHFULNESS_SYSTEM = """You check whether an ANSWER is grounded in the CONTEXT passages.
Split the answer into its factual claims. A claim is supported only if the context states or directly implies it.
Ignore citation markers like [1]. Sentences saying information is missing are not claims.
Reply with JSON only: {"supported_claims": <int>, "total_claims": <int>, "reason": "<one short sentence>"}"""


def _score(reply: str, key: str, allowed: set[float] | None = None) -> float | None:
    data = parse_json_object(reply)
    value = data.get(key) if data else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    if allowed is not None and value not in allowed:
        return None
    return min(1.0, max(0.0, value))


def correctness(judge: LLMClient, item: GoldenItem, answer: Answer) -> float | None:
    """0 / 0.5 / 1, or None if the judge's reply is unusable."""
    if item.type is QuestionType.NO_ANSWER:
        return 1.0 if answer.abstained else 0.0
    if answer.abstained:
        return 0.0  # abstaining on an answerable question is a miss
    system = CORRECTNESS_SYSTEM + (AMBIGUOUS_RULE if item.type is QuestionType.AMBIGUOUS else "")
    user = f"QUESTION:\n{item.question}\n\nGOLD ANSWER:\n{item.gold_answer}\n\nSYSTEM ANSWER:\n{answer.answer_text}"
    return _score(judge.complete(system, user), "score", allowed={0.0, 0.5, 1.0})


def faithfulness(judge: LLMClient, answer: Answer) -> float | None:
    """Share of claims grounded in the retrieved context. None when abstained or unusable."""
    if answer.abstained or not answer.answer_text.strip():
        return None
    user = f"CONTEXT:\n{format_context(answer.retrieved)}\n\nANSWER:\n{answer.answer_text}"
    data = parse_json_object(judge.complete(FAITHFULNESS_SYSTEM, user))
    if not data:
        return None
    supported, total = data.get("supported_claims"), data.get("total_claims")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (supported, total)):
        return None
    if total == 0:
        return 1.0  # nothing claimed, nothing unfaithful
    return min(1.0, supported / total)
