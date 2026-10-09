import pytest

from rag.dashboard.helpers import (
    chunk_table,
    citation_status,
    cited_markers,
    confidence_rows,
    format_score,
    linkify_citations,
    source_title,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Port 7420 [1].", "Port 7420 [[1]](#source-1)."),
        ("A [1][2].", "A [[1]](#source-1)[[2]](#source-2)."),
        ("A [1, 3].", "A [[1]](#source-1)[[3]](#source-3)."),
        ("No citations.", "No citations."),
        ("Array a[i] stays.", "Array a[i] stays."),
    ],
)
def test_linkify_citations(text, expected):
    assert linkify_citations(text) == expected


def test_linkify_uses_prefix_for_side_by_side():
    assert linkify_citations("A [2].", anchor_prefix="dense") == "A [[2]](#dense-2)."


def test_format_score():
    assert format_score(0.12345) == "0.12"
    assert format_score(0.12345, 3) == "0.123"
    assert format_score(None) == "—"


def test_citation_status():
    assert citation_status(True).startswith("✅")
    assert citation_status(False).startswith("❌")
    assert citation_status(None).startswith("⏳")


def test_confidence_rows_order_and_clamp():
    rows = confidence_rows({"retrieval": 0.9, "citation_coverage": 1.4, "completeness": -0.2, "composite": 0.5})
    assert rows == [("Retrieval", 0.9), ("Citation coverage", 1.0), ("Completeness", 0.0), ("Composite", 0.5)]
    assert confidence_rows({})[-1] == ("Composite", 0.0)


def test_source_title():
    assert source_title({"doc_id": "d"}) == "d"
    assert source_title({"doc_id": "d", "section_heading": "A > B", "page_number": 3}) == "d — A > B (p. 3)"


def test_cited_markers():
    assert cited_markers({"citations": [{"marker": 2}, {"marker": 1}, {"marker": 2}]}) == {1, 2}
    assert cited_markers({}) == set()


def test_chunk_table():
    retrieved = [
        {
            "chunk": {"doc_id": "d1", "section_heading": "S"},
            "score": 0.91234,
            "dense_rank": 2,
            "sparse_rank": None,
            "rerank_score": 0.91234,
        },
        {
            "chunk": {"doc_id": "d2", "section_heading": None},
            "score": 0.5,
            "dense_rank": None,
            "sparse_rank": 1,
            "rerank_score": None,
        },
    ]
    rows = chunk_table(retrieved)
    assert rows[0] == {"Rank": 1, "Document": "d1", "Section": "S", "Score": "0.912", "Dense rank": 2,
                       "BM25 rank": "—", "Rerank": "0.912"}  # fmt: skip
    assert rows[1]["Section"] == "" and rows[1]["BM25 rank"] == 1 and rows[1]["Rerank"] == "—"
