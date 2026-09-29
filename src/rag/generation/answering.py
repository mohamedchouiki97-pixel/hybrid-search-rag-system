"""The answer flow for already-retrieved chunks:

    retrieval confidence < threshold  ->  structured abstain (no LLM call)
    otherwise                         ->  generate
        model refused                 ->  structured abstain (no judge calls)
        otherwise                     ->  verify citations -> score confidence

Retrieval itself happens outside (service.py wires the retriever in), so this
module depends only on core.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.core.config import ConfidenceWeights, Settings
from rag.core.interfaces import CitationVerifier, Generator, LLMClient
from rag.core.models import Answer, RetrievedChunk
from rag.generation.abstain import REFUSAL_MISSING, build_abstain_answer, is_refusal, should_abstain
from rag.generation.citation_verifier import LLMCitationVerifier
from rag.generation.confidence import ConfidenceScorer, retrieval_confidence
from rag.generation.generator import LLMGenerator


class AnswerFlow:
    def __init__(
        self,
        generator: Generator,
        verifier: CitationVerifier,
        scorer: ConfidenceScorer,
        abstain_threshold: float = 0.3,
        weights: ConfidenceWeights | None = None,
    ) -> None:
        self.generator = generator
        self.verifier = verifier
        self.scorer = scorer
        self.abstain_threshold = abstain_threshold
        self.weights = weights or ConfidenceWeights()

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        retrieval = retrieval_confidence(chunks)
        if should_abstain(retrieval, self.abstain_threshold):
            return build_abstain_answer(question, chunks, retrieval, self.abstain_threshold, self.weights)
        generated = self.generator.answer(question, chunks)
        if is_refusal(generated.answer_text):
            return build_abstain_answer(
                question, chunks, retrieval, self.abstain_threshold, self.weights, missing=REFUSAL_MISSING
            )
        answer = self.verifier.verify(generated, chunks)
        return answer.model_copy(update={"confidence": self.scorer.score(answer)})


def build_answer_flow(settings: Settings, llm: LLMClient, judge: LLMClient | None = None) -> AnswerFlow:
    """Wired from config. The judge defaults to the same client as generation."""
    judge = judge or llm
    return AnswerFlow(
        generator=LLMGenerator(llm),
        verifier=LLMCitationVerifier(judge),
        scorer=ConfidenceScorer(judge, settings.confidence_weights),
        abstain_threshold=settings.abstain_threshold,
        weights=settings.confidence_weights,
    )
