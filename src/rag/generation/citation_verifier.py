"""Citation verifier: a judge LLM checks each claim against the passages it cites.

One judge call per claim, with all of that claim's cited passages together, so a
fact assembled from [1] and [3] is judged the way a reader would judge it. The
judge also says which passages it needed: a cited passage it did not need is
marked unsupported (an irrelevant citation), which keeps citation accuracy honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from rag.core.interfaces import LLMClient
from rag.core.models import Answer, Chunk, Citation, RetrievedChunk
from rag.generation.prompts import CITATION_JUDGE_SYSTEM, citation_judge_prompt, parse_json_object

NOT_RETRIEVED_REASON = "cited passage is not in the retrieved context"
NOT_NEEDED_REASON = "claim is supported, but not by this passage"


class LLMCitationVerifier:
    """Never raises on a bad judge reply: unparseable output or a failed call marks the
    claim's citations unsupported, with the reason."""

    def __init__(self, judge: LLMClient, max_workers: int = 8) -> None:
        self.judge = judge
        self.max_workers = max_workers

    def verify(self, answer: Answer, chunks: Sequence[RetrievedChunk]) -> Answer:
        if not answer.citations:
            return answer
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in chunks}
        claims: dict[str, list[int]] = {}  # claim text -> citation indexes, in order
        for i, c in enumerate(answer.citations):
            claims.setdefault(c.claim_text, []).append(i)

        checked: list[Citation] = list(answer.citations)
        jobs = list(claims.items())
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as pool:
            for updates in pool.map(lambda job: self._check_claim(job[0], [(i, checked[i]) for i in job[1]], by_id), jobs):
                for i, citation in updates.items():
                    checked[i] = citation
        return answer.model_copy(update={"citations": checked})

    def _check_claim(self, claim: str, cites: list[tuple[int, Citation]], by_id: dict[str, Chunk]) -> dict[int, Citation]:
        out: dict[int, Citation] = {}
        valid: list[tuple[int, Citation]] = []
        for i, c in cites:
            if c.chunk_id is None or c.chunk_id not in by_id:
                out[i] = c.model_copy(update={"verified": False, "judge_reason": c.judge_reason or NOT_RETRIEVED_REASON})
            else:
                valid.append((i, c))
        if not valid:
            return out

        passages: dict[int, str] = {}
        for _, c in valid:
            passages.setdefault(c.marker, by_id[c.chunk_id].text)  # type: ignore[index]
        try:
            reply = self.judge.complete(CITATION_JUDGE_SYSTEM, citation_judge_prompt(claim, sorted(passages.items())))
        except Exception as exc:
            return out | {i: c.model_copy(update={"verified": False, "judge_reason": f"judge call failed: {exc}"}) for i, c in valid}

        data = parse_json_object(reply)
        if data is None or not isinstance(data.get("supported"), bool):
            reason = f"unparseable judge output: {reply[:200]!r}"
            return out | {i: c.model_copy(update={"verified": False, "judge_reason": reason}) for i, c in valid}

        supported, reason = data["supported"], str(data.get("reason", ""))
        used = data.get("used")
        used_markers = {n for n in used if isinstance(n, int)} if isinstance(used, list) else None
        for i, c in valid:
            needed = used_markers is None or c.marker in used_markers  # no usable list: trust the verdict
            ok = supported and needed
            out[i] = c.model_copy(update={"verified": ok, "judge_reason": reason if ok or not supported else NOT_NEEDED_REASON})
        return out
