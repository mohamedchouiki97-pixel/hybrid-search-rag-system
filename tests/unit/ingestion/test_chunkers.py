import pytest

from rag.core.config import Settings
from rag.core.fakes import FakeEmbedder
from rag.core.interfaces import Chunker
from rag.core.models import ChunkStrategy, DocFormat, Document
from rag.ingestion.chunkers import (
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    breakpoints,
    get_chunker,
    make_chunker,
    parse_sections,
    split_units,
)
from rag.ingestion.loaders import iter_corpus_files, load_document


def make_doc(text: str, doc_id: str = "doc", **metadata) -> Document:
    return Document(doc_id=doc_id, source_path=f"{doc_id}.md", format=DocFormat.MD, raw_path=f"{doc_id}.md",
                    text=text, metadata=metadata)  # fmt: skip


def topic_embedder(text: str, keyword: str = "cat") -> FakeEmbedder:
    """2-d vectors: [1, 0] for units mentioning `keyword`, [0, 1] for the rest."""
    units = [text[s:e] for s, e in split_units(text)]
    return FakeEmbedder(dim=2, overrides={u: [1.0, 0.0] if keyword in u.lower() else [0.0, 1.0] for u in units})


LONG_TEXT = "# Guide\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))

SECTIONED = """# Guide

Intro paragraph.

## Install

Run the installer.

```python
# not a heading
def f():
    return 1
```

## Configure

#### Deep detail

Set the port.

### Advanced

Tune the lease.
"""


# ---------- structure parsing ----------


def test_parse_sections_breadcrumbs_and_fences():
    sections = parse_sections(SECTIONED)
    assert [s.heading for s in sections] == [
        None, "Guide", "Guide > Install", "Guide > Configure", "Guide > Configure > Advanced",
    ]  # fmt: skip
    install = sections[2]
    assert "# not a heading" in SECTIONED[install.body_start : install.end]


def test_parse_sections_resets_deeper_levels():
    text = "# A\n## B\n### C\n## D\ntext"
    assert [s.heading for s in parse_sections(text)][-1] == "A > D"


# ---------- properties shared by all strategies ----------


@pytest.fixture(params=list(ChunkStrategy))
def chunker(request) -> Chunker:
    return make_chunker(request.param, chunk_size=300, chunk_overlap=50, embedder=FakeEmbedder())


def test_chunker_satisfies_protocol(chunker):
    assert isinstance(chunker, Chunker)


def test_chunk_ids_are_stable_across_runs(chunker):
    doc = make_doc(LONG_TEXT)
    first = [c.chunk_id for c in chunker.chunk(doc)]
    again = make_chunker(chunker.strategy, chunk_size=300, chunk_overlap=50, embedder=FakeEmbedder()).chunk(doc)
    assert first == [c.chunk_id for c in again]
    assert len(set(first)) == len(first) > 1


def test_chunks_are_tagged_and_indexed(chunker):
    chunks = chunker.chunk(make_doc(LONG_TEXT, doc_id="guide/intro"))
    assert all(c.strategy is chunker.strategy for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "guide/intro" and c.source_path == "guide/intro.md" for c in chunks)
    assert all(c.char_count <= 300 for c in chunks)
    assert all(c.section_heading == "Guide" for c in chunks[1:])


def test_page_numbers_follow_page_offsets(chunker):
    page1, page2 = "First page text. " * 20, "Second page text. " * 20
    text = f"{page1.strip()}\n\n{page2.strip()}"
    doc = make_doc(text, page_offsets=[[0, 1], [len(page1.strip()) + 2, 2]])
    pages = [c.page_number for c in chunker.chunk(doc)]
    assert pages[0] == 1 and pages[-1] == 2
    assert pages == sorted(pages)


def test_no_page_numbers_without_offsets(chunker):
    assert all(c.page_number is None for c in chunker.chunk(make_doc(LONG_TEXT)))


# ---------- fixed ----------


def test_fixed_respects_size_and_overlap():
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = FixedChunker(chunk_size=100, chunk_overlap=20).chunk(make_doc(text))
    assert all(c.char_count == 100 for c in chunks[:-1])
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text[-20:] == b.text[:20]
    rebuilt = chunks[0].text + "".join(c.text[20:] for c in chunks[1:])
    assert rebuilt == text


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (100, -1)])
def test_fixed_rejects_bad_overlap(size, overlap):
    with pytest.raises(ValueError):
        FixedChunker(size, overlap)


# ---------- recursive ----------


def test_recursive_keeps_fitting_sections_whole():
    chunks = RecursiveChunker(chunk_size=800).chunk(make_doc(SECTIONED))
    assert [c.section_heading for c in chunks] == [
        "Guide", "Guide > Install", "Guide > Configure", "Guide > Configure > Advanced",
    ]  # fmt: skip
    install = chunks[1].text
    assert install.startswith("Guide > Install\n\nRun the installer.")
    assert "```python\n# not a heading\ndef f():\n    return 1\n```" in install  # indentation kept
    assert "#### Deep detail\n\nSet the port." in chunks[2].text  # h4 stays in the body


def test_recursive_never_mixes_sections_when_splitting():
    text = "# A\n\n" + "alpha " * 200 + "\n\n# B\n\n" + "beta " * 200
    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=50).chunk(make_doc(text))
    assert len(chunks) > 4
    for c in chunks:
        assert c.char_count <= 300
        assert ("alpha" in c.text) != ("beta" in c.text)
        assert c.text.startswith(f"{c.section_heading}\n\n")


def test_recursive_skips_empty_sections_and_unheaded_preamble():
    text = "Preamble line.\n\n# Title\n## Empty\n## Full\n\nBody."
    chunks = RecursiveChunker().chunk(make_doc(text))
    assert [(c.section_heading, c.text) for c in chunks] == [
        (None, "Preamble line."), ("Title > Full", "Title > Full\n\nBody."),
    ]  # fmt: skip


# ---------- semantic ----------

TOPICS = (
    "# Pets\n\nCats purr when happy. A cat sleeps all day. Cats chase string.\n\n"
    "Rockets need fuel to launch. Orbits are reached at high speed. Engines burn hydrogen."
)


def test_split_units_sentences_code_and_headings():
    text = "# Title\n\nFirst one. Second one.\n\n```py\nx = 1. y = 2\n\nz = 3\n```\n\n## Tail"
    units = [text[s:e] for s, e in split_units(text)]
    assert units == ["# Title\n\nFirst one.", "Second one.", "```py\nx = 1. y = 2\n\nz = 3\n```", "## Tail"]


def test_split_units_unclosed_fence_runs_to_end():
    text = "Intro.\n\n```py\nx = 1"
    assert [text[s:e] for s, e in split_units(text)] == ["Intro.", "```py\nx = 1"]


def test_breakpoints_bottom_percentile():
    assert breakpoints([0.9, 0.2, 0.8, 0.95, 0.85], 0.10) == {1}
    assert breakpoints([0.9, 0.2, 0.8, 0.1], 0.50) == {1, 3}
    assert breakpoints([], 0.10) == set()


def test_semantic_splits_at_planted_topic_change():
    embedder = topic_embedder(TOPICS)
    chunks = SemanticChunker(embedder, chunk_size=1000, min_chunk_chars=0).chunk(make_doc(TOPICS))
    assert [c.text for c in chunks] == [
        "# Pets\n\nCats purr when happy. A cat sleeps all day. Cats chase string.",
        "Rockets need fuel to launch. Orbits are reached at high speed. Engines burn hydrogen.",
    ]
    assert len(embedder.calls) == 1  # one batched embed call per document


def test_semantic_splits_oversized_groups_and_units():
    code = "```py\n" + "\n".join(f"x{i} = {i}" for i in range(60)) + "\n```"
    text = "Cats purr. " * 30 + "\n\n" + code
    chunks = SemanticChunker(FakeEmbedder(), chunk_size=200, min_chunk_chars=0).chunk(make_doc(text))
    assert all(c.char_count <= 200 for c in chunks)
    assert "".join(c.text for c in chunks).replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


def test_semantic_merges_tiny_chunks():
    text = "Cats purr. Rockets fly. Cats nap. Rockets burn."
    embedder = topic_embedder(text)
    def run(size, min_chars):
        chunker = SemanticChunker(embedder, chunk_size=size, breakpoint_percentile=0.99, min_chunk_chars=min_chars)
        return [c.text for c in chunker.chunk(make_doc(text))]

    assert len(run(200, 0)) == 4  # every sentence is its own group
    # a piece under 12 chars ("Cats purr.", "Cats nap.") is glued to its neighbour
    assert run(200, 12) == ["Cats purr. Rockets fly. Cats nap.", "Rockets burn."]
    # ...but never past chunk_size
    assert run(25, 12) == ["Cats purr. Rockets fly.", "Cats nap. Rockets burn."]


def test_semantic_empty_units_gives_no_chunks():
    assert SemanticChunker(FakeEmbedder()).chunk(make_doc("   \n\n  ")) == []


def test_semantic_rejects_bad_percentile():
    with pytest.raises(ValueError):
        SemanticChunker(FakeEmbedder(), breakpoint_percentile=1.0)


# ---------- factory ----------


@pytest.mark.parametrize(
    ("strategy", "cls"),
    [(ChunkStrategy.FIXED, FixedChunker), (ChunkStrategy.RECURSIVE, RecursiveChunker), (ChunkStrategy.SEMANTIC, SemanticChunker)],
)
def test_get_chunker_from_settings(settings: Settings, strategy, cls):
    s = settings.model_copy(update={"chunk_strategy": strategy})
    assert isinstance(get_chunker(s, embedder=FakeEmbedder()), cls)


def test_semantic_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        make_chunker("semantic")


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        make_chunker("bogus")


# ---------- fixture corpus ----------


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_fixture_corpus_chunks_keep_known_facts_findable(corpus_dir, known_facts, strategy):
    chunker = make_chunker(strategy, embedder=FakeEmbedder())
    chunks = [c for p in iter_corpus_files(corpus_dir) for c in chunker.chunk(load_document(p, root=corpus_dir))]
    assert {c.doc_id for c in chunks} == {"overview", "installation", "configuration", "errors", "operations"}
    for fact in known_facts["lookup"]:
        assert any(fact["answer"] in c.text for c in chunks), fact["question"]
    if strategy is ChunkStrategy.RECURSIVE:
        rare = known_facts["rare_token"][0]
        [hit] = [c for c in chunks if c.section_heading and c.section_heading.endswith(rare["section"])]
        assert rare["answer"] in hit.text
