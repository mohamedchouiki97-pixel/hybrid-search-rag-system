"""Checks on the shared test setup itself (conftest.py and fixtures)."""

import socket

import pytest

from rag.core.models import Document


def test_unit_marker_applied_by_folder(request):
    assert request.node.get_closest_marker("unit") is not None


def test_network_is_blocked_in_unit_tests():
    with pytest.raises(RuntimeError, match="unit tests may not"):
        socket.create_connection(("example.com", 80), timeout=1)
    with pytest.raises(RuntimeError, match="unit tests may not"):
        socket.socket().connect(("93.184.215.14", 80))


def test_localhost_is_allowed():
    a, b = socket.socketpair()  # uses a loopback connect on Windows; asyncio needs this
    a.close()
    b.close()


def test_fixture_corpus_has_five_markdown_docs(corpus_dir):
    docs = sorted(p.name for p in corpus_dir.glob("*.md"))
    assert len(docs) == 5
    for p in corpus_dir.glob("*.md"):
        assert p.read_text(encoding="utf-8").startswith("# ")


def test_known_facts_point_at_real_text(corpus_dir, known_facts):
    assert {"lookup", "rare_token", "multi_hop", "no_answer"} <= set(known_facts)
    for fact in known_facts["lookup"] + known_facts["rare_token"]:
        text = (corpus_dir / fact["doc"]).read_text(encoding="utf-8")
        assert f"## {fact['section']}" in text
    rare = known_facts["rare_token"][0]
    hits = [p.name for p in corpus_dir.glob("*.md") if rare["token"] in p.read_text(encoding="utf-8")]
    assert hits == [rare["doc"]]


def test_no_answer_facts_are_absent(corpus_dir):
    corpus = " ".join(p.read_text(encoding="utf-8").lower() for p in corpus_dir.glob("*.md"))
    assert "kafka" not in corpus
    assert "license" not in corpus and "price" not in corpus


def test_document_model_accepts_fixture(corpus_dir):
    path = corpus_dir / "overview.md"
    doc = Document(doc_id="overview", source_path=str(path), format="md", raw_path=str(path), text=path.read_text(encoding="utf-8"))
    assert "7420" in doc.text
