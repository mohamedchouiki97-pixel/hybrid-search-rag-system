"""Citation verifier: a judge LLM checks that each cited passage supports its claim."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from rag.core.interfaces import LLMClient
from rag.core.models import Answer, Citation, RetrievedChunk
from rag.generation.prompts import CITATION_JUDGE_SYSTEM, citation_judge_prompt, parse_json_object

NOT_RETRIEVED_REASON = "cited passage is not in the retrieved context"


class LLMCitationVerifier:
    """One judge call per citation, run concurrently. Never raises on a bad judge reply:
    unparseable output or a failed call marks the citation unsupported, with the reason."""

    def __init__(self, judge: LLMClient, max_workers: int = 8) -> None:
        self.judge = judge
        self.max_workers = max_workers

    def verify(self, answer: Answer, chunks: Sequence[RetrievedChunk]) -> Answer:
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in chunks}

        def check(citation: Citation) -> Citation:
            if citation.chunk_id is None or citation.chunk_id not in by_id:
                reason = citation.judge_reason or NOT_RETRIEVED_REASON
                return citation.model_copy(update={"verified": False, "judge_reason": reason})
            try:
                reply = self.judge.complete(
                    CITATION_JUDGE_SYSTEM, citation_judge_prompt(citation.claim_text, by_id[citation.chunk_id].text)
                )
            except Exception as exc:
                return citation.model_copy(update={"verified": False, "judge_reason": f"judge call failed: {exc}"})
            verified, reason = _read_verdict(reply)
            return citation.model_copy(update={"verified": verified, "judge_reason": reason})

        if not answer.citations:
            return answer
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(answer.citations))) as pool:
            checked = list(pool.map(check, answer.citations))  # map keeps the original order
        return answer.model_copy(update={"citations": checked})


def _read_verdict(reply: str) -> tuple[bool, str]:
    data = parse_json_object(reply)
    if data is None or not isinstance(data.get("supported"), bool):
        return False, f"unparseable judge output: {reply[:200]!r}"
    return data["supported"], str(data.get("reason", ""))
