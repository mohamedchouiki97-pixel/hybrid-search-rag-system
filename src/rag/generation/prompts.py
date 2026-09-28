"""Prompts for answering and judging, plus a tolerant JSON parser for judge replies."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from rag.core.models import RetrievedChunk

NO_ANSWER_SENTENCE = "The provided documents do not answer this question."

ANSWER_SYSTEM = f"""You answer questions using ONLY the numbered context passages you are given.

Rules:
- End every factual sentence with the numbers of the passages that support it, like [1] or [2][3].
- Cite only passage numbers that appear in the context. Never cite a passage that does not support the sentence.
- Do not use outside knowledge, even if you know the answer.
- If the context does not answer the question, reply exactly: "{NO_ANSWER_SENTENCE}"
- If the context answers only part of the question, answer that part and say which part is missing.
- Be concise and direct."""

CITATION_JUDGE_SYSTEM = """You check whether a PASSAGE supports a CLAIM.
The claim is supported only if the passage states it or directly implies it. Partial or related information is not support.
Reply with JSON only, no other text: {"supported": true or false, "reason": "<one short sentence>"}"""

COMPLETENESS_JUDGE_SYSTEM = """You rate how completely an ANSWER addresses every part of a QUESTION.
Judge coverage of the question's parts, not whether the facts are true.
Reply with JSON only, no other text: {"score": <number from 0 to 1>, "reason": "<one short sentence>"}
1 means every part is addressed; 0 means nothing is addressed or the answer declines to answer."""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered passages [1]..[n], each with its source and section so the model can cite them."""
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        where = c.source_path + (f" | {c.section_heading}" if c.section_heading else "")
        if c.page_number is not None:
            where += f" | page {c.page_number}"
        blocks.append(f"[{i}] (source: {where})\n{c.text}")
    return "\n\n".join(blocks)


def answer_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    return f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer with citations:"


def citation_judge_prompt(claim: str, passage: str) -> str:
    return f"CLAIM:\n{claim}\n\nPASSAGE:\n{passage}"


def completeness_judge_prompt(question: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nANSWER TO RATE:\n{answer}"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in text (tolerates ```json fences and chatter). None if there is none."""
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
