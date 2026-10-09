import json

import pytest

from rag.evaluation.golden import (
    QA_PATH,
    GoldenFileError,
    GoldenItem,
    QuestionType,
    load_catalog,
    load_golden,
    validate_golden,
)

CATALOG = {"sections": [{"doc_id": "d", "section": "A"}, {"doc_id": "d", "section": "A > B"}]}


def item(i, qtype="lookup", sections=({"doc_id": "d", "section": "A"},)):
    return GoldenItem.model_validate(
        {"id": f"q{i:03d}", "type": qtype, "question": f"question {i}?", "gold_answer": "answer",
         "gold_chunk_sections": list(sections)}
    )  # fmt: skip


def full_set():
    types = ["lookup", "multi_hop", "no_answer", "ambiguous"]
    return [
        item(i, types[i % 4], () if types[i % 4] == "no_answer" else ({"doc_id": "d", "section": "A"},))
        for i in range(50)
    ]


# ---------- the real golden file ----------


def test_real_golden_file_is_valid():
    items = load_golden()
    assert validate_golden(items, load_catalog()) == []
    assert len(items) >= 50
    assert {i.type for i in items} == set(QuestionType)


def test_real_golden_lines_all_parse():
    for n, line in enumerate(QA_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        assert isinstance(json.loads(line), dict), f"line {n}"


# ---------- validator ----------


def test_valid_set_passes():
    assert validate_golden(full_set(), CATALOG) == []


def test_too_few_questions():
    assert validate_golden(full_set()[:49], CATALOG) == ["49 questions; at least 50 required"]


def test_duplicate_ids():
    items = full_set()
    items[1] = items[1].model_copy(update={"id": items[0].id})
    assert any("duplicate id 'q000'" in p for p in validate_golden(items, CATALOG))


def test_missing_type():
    items = [i for i in full_set() if i.type is not QuestionType.AMBIGUOUS]
    items += [item(100 + n) for n in range(50 - len(items))]
    assert "missing question types: ['ambiguous']" in validate_golden(items, CATALOG)


def test_unknown_section_and_missing_gold():
    items = full_set()
    items[0] = item(0, sections=({"doc_id": "d", "section": "A > C"},))
    items[4] = item(4, sections=())
    problems = validate_golden(items, CATALOG)
    assert "q000: gold section not in catalog: d | A > C" in problems
    assert "q004: lookup question needs gold sections" in problems


# ---------- loading ----------


def write(tmp_path, lines):
    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_load_reports_line_of_bad_json(tmp_path):
    good = json.dumps(item(1).model_dump(mode="json"))
    with pytest.raises(GoldenFileError, match="line 2"):
        load_golden(write(tmp_path, [good, "{not json"]))


@pytest.mark.parametrize(
    "row",
    [
        {"id": "q1", "type": "trivia", "question": "q", "gold_answer": "a"},  # unknown type
        {"id": "q1", "type": "lookup", "question": "", "gold_answer": "a"},  # empty question
        {"id": "q1", "type": "lookup", "question": "q", "gold_answer": "a", "extra": 1},  # unknown field
        {"id": "q1", "type": "lookup", "question": "q", "gold_answer": "a",
         "gold_chunk_sections": [{"doc_id": "d"}]},  # section missing
    ],
)  # fmt: skip
def test_load_rejects_bad_rows(tmp_path, row):
    with pytest.raises(GoldenFileError, match="line 1"):
        load_golden(write(tmp_path, [json.dumps(row)]))


def test_load_rejects_blank_line(tmp_path):
    good = json.dumps(item(1).model_dump(mode="json"))
    with pytest.raises(GoldenFileError, match="line 2: blank"):
        load_golden(write(tmp_path, [good, "", good]))
