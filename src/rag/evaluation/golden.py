"""The golden dataset: loading, and validation against the section catalog."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

GOLDEN_DIR = Path(__file__).parent / "golden"
QA_PATH = GOLDEN_DIR / "qa.jsonl"
SECTIONS_PATH = GOLDEN_DIR / "sections.json"
MIN_QUESTIONS = 50


class QuestionType(StrEnum):
    LOOKUP = "lookup"
    MULTI_HOP = "multi_hop"
    NO_ANSWER = "no_answer"
    AMBIGUOUS = "ambiguous"


class GoldSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    section: str = Field(min_length=1)


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: QuestionType
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    gold_chunk_sections: list[GoldSection] = Field(default_factory=list)


class GoldenFileError(ValueError):
    pass


def load_golden(path: str | Path = QA_PATH) -> list[GoldenItem]:
    """Parse qa.jsonl. Raises GoldenFileError naming the line of the first bad row."""
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise GoldenFileError(f"line {n}: blank line")
        try:
            items.append(GoldenItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GoldenFileError(f"line {n}: {exc}") from exc
    return items


def load_catalog(path: str | Path = SECTIONS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_golden(items: list[GoldenItem], catalog: dict, min_questions: int = MIN_QUESTIONS) -> list[str]:
    """Every problem found (empty list = valid)."""
    problems: list[str] = []
    if len(items) < min_questions:
        problems.append(f"{len(items)} questions; at least {min_questions} required")
    for qid, count in Counter(i.id for i in items).items():
        if count > 1:
            problems.append(f"duplicate id {qid!r} ({count}x)")
    missing_types = set(QuestionType) - {i.type for i in items}
    if missing_types:
        problems.append(f"missing question types: {sorted(t.value for t in missing_types)}")

    valid = {(s["doc_id"], s["section"]) for s in catalog["sections"]}
    for item in items:
        for g in item.gold_chunk_sections:
            if (g.doc_id, g.section) not in valid:
                problems.append(f"{item.id}: gold section not in catalog: {g.doc_id} | {g.section}")
        if item.type in (QuestionType.LOOKUP, QuestionType.MULTI_HOP) and not item.gold_chunk_sections:
            problems.append(f"{item.id}: {item.type.value} question needs gold sections")
    return problems
