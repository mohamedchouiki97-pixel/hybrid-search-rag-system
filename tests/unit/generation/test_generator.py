import pytest

from rag.core.fakes import FakeLLM
from rag.core.interfaces import Generator
from rag.core.models import RetrievedChunk
from rag.generation.generator import UNMATCHED_REASON, Claim, LLMGenerator, extract_claims, parse_citations
from rag.generation.prompts import ANSWER_SYSTEM, answer_prompt, format_context


@pytest.fixture
def chunks(make_chunk):
    texts = ["The broker listens on port 7420.", "Retention is 72 hours.", "Backups are incremental."]
    return [
        RetrievedChunk(chunk=make_chunk(t, doc_id=f"d{i}", section_heading=f"Sec {i}"), score=0.9 - i / 10)
        for i, t in enumerate(texts)
    ]


# ---------- prompts ----------


def test_prompt_contains_every_chunk_numbered(chunks):
    prompt = answer_prompt("What port?", chunks)
    for i, rc in enumerate(chunks, start=1):
        assert f"[{i}] (source: {rc.chunk.source_path} | {rc.chunk.section_heading})\n{rc.chunk.text}" in prompt
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]")
    assert "[4]" not in prompt
    assert "Question: What port?" in prompt


def test_context_includes_page_numbers(make_chunk):
    ctx = format_context([RetrievedChunk(chunk=make_chunk("t", page_number=4), score=1)])
    assert "| page 4)" in ctx


def test_system_prompt_rules():
    for rule in ("ONLY", "[1]", "outside knowledge", "do not answer"):
        assert rule in ANSWER_SYSTEM


# ---------- claim / citation parsing ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Port is 7420 [1].", [Claim("Port is 7420.", (1,))]),
        ("Port is 7420 [1][2].", [Claim("Port is 7420.", (1, 2))]),
        ("Port is 7420 [1, 2].", [Claim("Port is 7420.", (1, 2))]),
        ("Port is 7420. [2] Kept 72 hours [1].", [Claim("Port is 7420.", (2,)), Claim("Kept 72 hours.", (1,))]),
        ("No citations here.", [Claim("No citations here.", ())]),
        ("Run main.py on version 3.5 [1].", [Claim("Run main.py on version 3.5.", (1,))]),
        ("Found:\n- Port 7420 [1]\n- Dashboard 7421 [2]", [Claim("Port 7420", (1,)), Claim("Dashboard 7421", (2,))]),
        ("1. Port 7420 [1]\n2. Kept 72 hours [2]", [Claim("Port 7420", (1,)), Claim("Kept 72 hours", (2,))]),
        ("Dup [1][1, 2].", [Claim("Dup.", (1, 2))]),
        ("", []),
    ],
)
def test_extract_claims(text, expected):
    assert extract_claims(text) == expected


def test_uncited_sentences_group_with_next_cited_sentence():
    text = "Order matters. Declare /users/me first. Otherwise it matches the other route [1][2]."
    assert extract_claims(text) == [
        Claim("Order matters. Declare /users/me first. Otherwise it matches the other route.", (1, 2), sentences=3)
    ]


def test_grouping_stops_at_paragraph_breaks():
    text = "Intro sentence.\n\nFirst fact [1]. Loose end.\n\nSecond fact [2]."
    assert extract_claims(text) == [
        Claim("Intro sentence.", ()),
        Claim("First fact.", (1,)),
        Claim("Loose end.", ()),
        Claim("Second fact.", (2,)),
    ]


CODE_ANSWER = """You can declare an optional query parameter by setting its default value to `None`. For example:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None):
    return {"item_id": item_id}
```

In this case, `q` is optional and defaults to `None` [1]."""


def test_code_blocks_are_not_claims_and_do_not_split_paragraphs():
    # Regression: each code line used to count as an uncited claim (coverage 1/7 = 0.14).
    assert extract_claims(CODE_ANSWER) == [
        Claim(
            "You can declare an optional query parameter by setting its default value to `None`. "
            "In this case, `q` is optional and defaults to `None`.",
            (1,),
            sentences=2,
        )
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Intro.\n\n~~~\nx = 1. y = 2\n~~~\n\nFact [1].", [Claim("Intro. Fact.", (1,), sentences=2)]),  # tilde fence
        ("Fact [1].\n\n```py\nunclosed = True", [Claim("Fact.", (1,))]),  # unclosed fence runs to the end
        ("A [1].\n\n```\ncode\n```\n\nB.\n\nC [2].", [Claim("A.", (1,)), Claim("B.", ()), Claim("C.", (2,))]),
        ("Only code:\n\n```\nprint(1)\n```", []),
    ],
)
def test_code_block_edge_cases(text, expected):
    assert extract_claims(text) == expected


def test_parse_citations_gives_grouped_claim_text(chunks):
    cites = parse_citations("Port is 7420. It is the default [1].", chunks)
    assert [(c.marker, c.claim_text) for c in cites] == [(1, "Port is 7420. It is the default.")]


def test_parse_citations_maps_markers_to_chunks(chunks):
    cites = parse_citations("Port is 7420 [1]. Kept 72 hours [2][3].", chunks)
    assert [(c.marker, c.chunk_id, c.claim_text) for c in cites] == [
        (1, chunks[0].chunk.chunk_id, "Port is 7420."),
        (2, chunks[1].chunk.chunk_id, "Kept 72 hours."),
        (3, chunks[2].chunk.chunk_id, "Kept 72 hours."),
    ]
    assert all(c.verified is None for c in cites)


@pytest.mark.parametrize("marker", [0, 4, 99])
def test_marker_without_chunk_is_flagged(chunks, marker):
    [cite] = parse_citations(f"Invented fact [{marker}].", chunks)
    assert (cite.chunk_id, cite.verified, cite.judge_reason) == (None, False, UNMATCHED_REASON)


def test_no_citations(chunks):
    assert parse_citations("The provided documents do not answer this question.", chunks) == []


# ---------- generator ----------


def test_generator_calls_llm_and_parses(chunks):
    llm = FakeLLM({"Question: What port": "  The broker uses port 7420 [1].  "})
    answer = LLMGenerator(llm).answer("What port?", chunks)
    assert answer.answer_text == "The broker uses port 7420 [1]."
    assert [c.chunk_id for c in answer.citations] == [chunks[0].chunk.chunk_id]
    assert answer.retrieved == chunks
    assert answer.abstained is False
    [(system, user)] = llm.calls
    assert system == ANSWER_SYSTEM and "[3]" in user


def test_generator_satisfies_protocol():
    assert isinstance(LLMGenerator(FakeLLM()), Generator)
