"""Generator: prompt the LLM with numbered context and parse its [n] citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.core.interfaces import LLMClient
from rag.core.models import Answer, Citation, RetrievedChunk
from rag.generation.prompts import ANSWER_SYSTEM, answer_prompt

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MARKERS = r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])*"
# A sentence runs to . ! ? (plus any citation markers right after it) or to the end
# of the line. A dot inside a token ("3.5", "main.py") does not end a sentence.
_SENTENCE_RE = re.compile(rf"[^\s](?:[^.!?\n]|[.!?](?=[^\s]))*(?:[.!?]+{_MARKERS}|$)", re.MULTILINE)

_BULLET_RE = re.compile(r"^[-*•]\s+")
_LIST_NUMBER_RE = re.compile(r"\d+[.)]?")  # "1." split off a numbered list item

_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")

UNMATCHED_REASON = "no retrieved passage has this number"


@dataclass(frozen=True)
class Claim:
    text: str  # sentence(s) with the citation markers removed
    markers: tuple[int, ...]  # distinct markers, in order of appearance
    sentences: int = 1  # how many sentences this claim covers


def _sentences(answer_text: str) -> list[tuple[int, str, tuple[int, ...]]]:
    """(paragraph number, sentence without markers, markers) for each sentence."""
    breaks = [m.end() for m in _PARAGRAPH_BREAK_RE.finditer(answer_text)]
    out = []
    for m in _SENTENCE_RE.finditer(answer_text):
        sentence = m.group(0).strip()
        markers: list[int] = []
        for group in _MARKER_RE.findall(sentence):
            for n in (int(x) for x in group.split(",")):
                if n not in markers:
                    markers.append(n)
        text = " ".join(_MARKER_RE.sub("", sentence).split())
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)  # "7420 ." -> "7420."
        text = _BULLET_RE.sub("", text)
        if text and not text.endswith(":") and not _LIST_NUMBER_RE.fullmatch(text):
            out.append((sum(1 for b in breaks if b <= m.start()), text, tuple(markers)))
    return out


def extract_claims(answer_text: str) -> list[Claim]:
    """Split an answer into claims and the markers that back each one.

    Handles [1], [1][2] and [1, 2]. Lead-in lines ending with ":" are not claims.

    Models often cite once at the end of a paragraph: "A. B. C [1][2]." So uncited
    sentences are grouped with the next cited sentence in the same paragraph into one
    claim, "A. B. C.", which the judge then checks as a whole against [1] and [2].
    Uncited sentences with no citation after them in their paragraph stay uncited claims.
    """
    claims: list[Claim] = []
    pending: list[str] = []
    paragraph = None

    def flush() -> None:
        claims.extend(Claim(text=s, markers=()) for s in pending)
        pending.clear()

    for para, text, markers in _sentences(answer_text):
        if para != paragraph:
            flush()
            paragraph = para
        if not markers:
            pending.append(text)
            continue
        claims.append(Claim(text=" ".join([*pending, text]), markers=markers, sentences=len(pending) + 1))
        pending.clear()
    flush()
    return claims


def parse_citations(answer_text: str, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """One Citation per (claim, marker). Markers with no matching chunk are flagged unverified."""
    citations: list[Citation] = []
    for claim in extract_claims(answer_text):
        for n in claim.markers:
            if 1 <= n <= len(chunks):
                citations.append(Citation(marker=n, chunk_id=chunks[n - 1].chunk.chunk_id, claim_text=claim.text))
            else:
                citations.append(
                    Citation(marker=n, chunk_id=None, claim_text=claim.text, verified=False, judge_reason=UNMATCHED_REASON)
                )
    return citations


class LLMGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Answer:
        text = self.llm.complete(ANSWER_SYSTEM, answer_prompt(question, chunks)).strip()
        return Answer(
            question=question,
            answer_text=text,
            citations=parse_citations(text, chunks),
            retrieved=list(chunks),
        )
