"""Confidence: how much to trust an answer, on three dimensions plus a weighted composite.

- retrieval:          did we find strongly relevant passages? (top chunk's cross-encoder
                      probability, in both hybrid and dense mode, so one threshold fits both)
- citation_coverage:  share of answer sentences inside a claim with a verified citation
                      (uncited sentences count as unbacked)
- completeness:       judge's view of whether every part of the question was addressed
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.core.config import ConfidenceWeights
from rag.core.interfaces import LLMClient
from rag.core.models import Answer, Confidence, RetrievedChunk
from rag.generation.generator import extract_claims
from rag.generation.prompts import COMPLETENESS_JUDGE_SYSTEM, completeness_judge_prompt, parse_json_object


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def retrieval_confidence(chunks: Sequence[RetrievedChunk]) -> float:
    """Best chunk's rerank score (0..1), falling back to raw scores for chunks never reranked.

    The best, not the first: in dense mode chunks stay in dense order, so the most
    relevant passage (by cross-encoder) can be at rank 3. In hybrid mode it is the first.
    """
    if not chunks:
        return 0.0
    return _clamp(max(rc.rerank_score if rc.rerank_score is not None else rc.score for rc in chunks))


def citation_coverage(answer: Answer) -> float:
    claims = extract_claims(answer.answer_text)
    if answer.abstained or not claims:
        return 0.0
    backed = {c.claim_text for c in answer.citations if c.verified}
    return sum(claim.sentences for claim in claims if claim.text in backed) / sum(claim.sentences for claim in claims)


def composite(retrieval: float, coverage: float, completeness: float, weights: ConfidenceWeights) -> float:
    w = weights.normalized()
    return _clamp(w["retrieval"] * retrieval + w["citation_coverage"] * coverage + w["completeness"] * completeness)


class ConfidenceScorer:
    def __init__(self, judge: LLMClient, weights: ConfidenceWeights | None = None) -> None:
        self.judge = judge
        self.weights = weights or ConfidenceWeights()

    def completeness(self, question: str, answer_text: str) -> float:
        """Judge score in 0..1; 0 if the judge fails or replies with something unusable."""
        try:
            reply = self.judge.complete(COMPLETENESS_JUDGE_SYSTEM, completeness_judge_prompt(question, answer_text))
        except Exception:
            return 0.0
        data = parse_json_object(reply)
        score = data.get("score") if data else None
        if isinstance(score, bool) or not isinstance(score, int | float):
            return 0.0
        return _clamp(float(score))

    def score(self, answer: Answer) -> Confidence:
        r = retrieval_confidence(answer.retrieved)
        cov = citation_coverage(answer)
        comp = self.completeness(answer.question, answer.answer_text)
        return Confidence(
            retrieval=r, citation_coverage=cov, completeness=comp, composite=composite(r, cov, comp, self.weights)
        )
