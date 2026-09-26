from pathlib import Path

import pytest

from rag.core.interfaces import Loader
from rag.core.models import DocFormat
from rag.ingestion.loaders import (
    HtmlLoader,
    IngestionError,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
    clean_markdown,
    detect_format,
    get_loader,
    iter_corpus_files,
    load_document,
    load_saved_document,
    make_doc_id,
    normalize_text,
    save_document,
)

# ---------- normalization ----------


def test_normalize_text():
    raw = "Café line  \r\nnext\r\r\n\n\n\nlast\x00  "
    assert normalize_text(raw) == "Café line\nnext\n\nlast"


# ---------- markdown ----------


@pytest.fixture
def md_doc(loader_fixtures_dir):
    return MarkdownLoader(root=loader_fixtures_dir).load(loader_fixtures_dir / "sample.md")


def test_markdown_metadata(md_doc):
    assert md_doc.doc_id == "sample"
    assert md_doc.source_path == "sample.md"
    assert md_doc.format is DocFormat.MD
    assert md_doc.metadata["title"] == "Sample Guide"


def test_markdown_removes_anchors_and_admonition_markers(md_doc):
    assert "{ #" not in md_doc.text
    assert "///" not in md_doc.text
    assert "# Sample Guide\n" in md_doc.text
    assert "Tip:\n\nUse a virtual environment." in md_doc.text
    assert "Note: Technical Details" in md_doc.text


def test_markdown_strips_html_but_keeps_text_and_inline_code(md_doc):
    assert "with API markup, an entity & inline code `<form>`." in md_doc.text
    assert "<div" not in md_doc.text and "<abbr" not in md_doc.text


def test_markdown_keeps_code_verbatim(md_doc):
    assert "# this comment is not a heading" in md_doc.text
    assert 'html = "<h1>kept</h1>"' in md_doc.text


def test_markdown_cleans_terminal_colouring(md_doc):
    assert "$ nimbus start" in md_doc.text
    assert "<font" not in md_doc.text


def test_markdown_drops_mkdocstrings_directive(md_doc):
    assert ":::" not in md_doc.text and "openapi_version" not in md_doc.text
    assert "## Reference\n\nAfter the directive." in md_doc.text


def test_clean_markdown_handles_indented_and_tab_blocks():
    text = "//// tab | Python 3.10+\n\nbody\n\n////\n\n    /// note\n    indented\n    ///\n"
    assert clean_markdown(text) == "Tab: Python 3.10+\n\nbody\n\nNote:\n    indented"


def test_clean_markdown_longer_fence_is_not_closed_by_shorter():
    text = "````md\n```python\n# inner\n```\n# still code\n````\n# Real"
    cleaned = clean_markdown(text)
    assert cleaned.endswith("````\n# Real")
    assert "# still code" in cleaned


def test_markdown_title_ignores_code_comments(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("```python\n# not a title\n```\n\n# Real Title\n", encoding="utf-8")
    assert MarkdownLoader().load(p).metadata["title"] == "Real Title"


def test_markdown_only_directive_raises(tmp_path):
    p = tmp_path / "stub.md"
    p.write_text("::: fastapi.FastAPI\n    options:\n        members: []\n", encoding="utf-8")
    with pytest.raises(IngestionError, match="no text content"):
        MarkdownLoader().load(p)


# ---------- text ----------


def test_text_loader(loader_fixtures_dir):
    doc = TextLoader().load(loader_fixtures_dir / "sample.txt")
    assert doc.format is DocFormat.TXT
    assert doc.text == "Plain text notes.\n\nSecond paragraph after many blank lines."
    assert doc.metadata == {}


def test_text_loader_handles_crlf_and_latin1(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("café\r\nline two\r\n".encode("latin-1"))
    assert TextLoader().load(p).text == "café\nline two"


def test_text_loader_strips_utf8_bom(tmp_path):
    p = tmp_path / "bom.txt"
    p.write_bytes("﻿hello".encode())
    assert TextLoader().load(p).text == "hello"


# ---------- html ----------


@pytest.fixture
def html_doc(loader_fixtures_dir):
    return HtmlLoader().load(loader_fixtures_dir / "sample.html")


def test_html_title_and_format(html_doc):
    assert html_doc.format is DocFormat.HTML
    assert html_doc.metadata["title"] == "Sample Page"


def test_html_converts_structure_to_markdown(html_doc):
    assert "# Main Title" in html_doc.text
    assert "## Details" in html_doc.text
    assert "First paragraph with bold text that wraps." in html_doc.text
    assert "- Item one\n- Item two" in html_doc.text
    assert '```\n# a code comment\nprint("hi")\n```' in html_doc.text
    assert "Line one\nLine two" in html_doc.text


def test_html_drops_script_style_nav(html_doc):
    for noise in ("drop me", "color: red", "Home"):
        assert noise not in html_doc.text


def test_html_title_falls_back_to_first_heading(tmp_path):
    p = tmp_path / "x.html"
    p.write_text("<body><h1>Heading Title</h1><p>text</p></body>", encoding="utf-8")
    assert HtmlLoader().load(p).metadata["title"] == "Heading Title"


# ---------- pdf ----------


def test_pdf_text_and_page_offsets(make_pdf):
    path = make_pdf(["Page one about leases.", "", "Page three about backups."])
    doc = PdfLoader().load(path)
    assert doc.format is DocFormat.PDF
    assert doc.metadata["page_count"] == 3
    offsets = doc.metadata["page_offsets"]
    assert [page for _, page in offsets] == [1, 3]  # empty page 2 skipped
    assert doc.text[offsets[0][0] :].startswith("Page one")
    assert doc.text[offsets[1][0] :].startswith("Page three")


def test_pdf_without_text_raises(make_pdf):
    with pytest.raises(IngestionError, match="no extractable text"):
        PdfLoader().load(make_pdf(["", ""]))


def test_corrupt_pdf_raises(tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(IngestionError, match="could not read PDF"):
        PdfLoader().load(p)


# ---------- errors shared by all loaders ----------


@pytest.mark.parametrize("loader_cls", [MarkdownLoader, TextLoader, HtmlLoader, PdfLoader])
@pytest.mark.parametrize("content", [b"", b"  \n\t\n"])
def test_empty_file_raises(tmp_path, loader_cls, content):
    p = tmp_path / "empty.bin"
    p.write_bytes(content)
    with pytest.raises(IngestionError, match="empty file"):
        loader_cls().load(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(IngestionError, match="file not found"):
        TextLoader().load(tmp_path / "nope.txt")


def test_ingestion_error_is_value_error():
    assert issubclass(IngestionError, ValueError)


# ---------- registry ----------


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("a.md", DocFormat.MD), ("a.MARKDOWN", DocFormat.MD), ("a.txt", DocFormat.TXT),
     ("a.htm", DocFormat.HTML), ("a.html", DocFormat.HTML), ("a.PDF", DocFormat.PDF)],
)  # fmt: skip
def test_detect_format(name, fmt):
    assert detect_format(name) is fmt


@pytest.mark.parametrize("name", ["a.docx", "Makefile"])
def test_unsupported_format_raises_clear_error(name):
    with pytest.raises(IngestionError, match=r"unsupported file type .*; supported: .*.html"):
        detect_format(name)


def test_get_loader_returns_protocol_instance(loader_fixtures_dir):
    loader = get_loader("x.html")
    assert isinstance(loader, HtmlLoader) and isinstance(loader, Loader)
    assert load_document(loader_fixtures_dir / "sample.txt").doc_id == "sample"


# ---------- doc ids and corpus walking ----------


def test_make_doc_id(tmp_path):
    root = tmp_path / "docs"
    assert make_doc_id(root / "tutorial" / "first-steps.md", root) == "tutorial/first-steps"
    assert make_doc_id(tmp_path / "elsewhere" / "x.md", root) == "x"
    assert make_doc_id(tmp_path / "upload.pdf") == "upload"


def test_doc_id_and_source_path_are_relative_to_root(tmp_path):
    root = tmp_path / "docs"
    (root / "guide").mkdir(parents=True)
    p = root / "guide" / "intro.md"
    p.write_text("# Intro\n\ntext", encoding="utf-8")
    doc = load_document(p, root=root)
    assert (doc.doc_id, doc.source_path) == ("guide/intro", "guide/intro.md")


def test_iter_corpus_files_filters_and_sorts(tmp_path):
    for rel in ["b.md", "a.txt", "sub/c.html", "reference/stub.md", "sub/reference/x.md", "LICENSE", "img.png"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    found = [p.relative_to(tmp_path).as_posix() for p in iter_corpus_files(tmp_path)]
    assert found == ["a.txt", "b.md", "sub/c.html"]
    assert len(list(iter_corpus_files(tmp_path, exclude_dirs=()))) == 5


# ---------- storage ----------


def test_save_and_reload_document(tmp_path, loader_fixtures_dir):
    doc = load_document(loader_fixtures_dir / "sample.md", root=loader_fixtures_dir)
    saved = save_document(doc, tmp_path / "documents")
    raw = Path(saved.raw_path)
    assert raw == tmp_path / "documents" / "sample" / "raw.md"
    assert raw.read_bytes() == (loader_fixtures_dir / "sample.md").read_bytes()
    assert (raw.parent / "text.md").read_text(encoding="utf-8") == doc.text
    assert load_saved_document(tmp_path / "documents", "sample") == saved


def test_save_is_idempotent_when_raw_already_stored(tmp_path, loader_fixtures_dir):
    doc = load_document(loader_fixtures_dir / "sample.txt")
    saved = save_document(doc, tmp_path)
    assert save_document(saved, tmp_path) == saved


def test_nested_doc_id_is_saved_in_subfolder(tmp_path, loader_fixtures_dir):
    doc = load_document(loader_fixtures_dir / "sample.txt").model_copy(update={"doc_id": "guide/intro"})
    saved = save_document(doc, tmp_path)
    assert Path(saved.raw_path) == tmp_path / "guide" / "intro" / "raw.txt"


@pytest.mark.parametrize("bad_id", ["../escape", "..", "a/../../b"])
def test_save_rejects_doc_id_outside_folder(tmp_path, loader_fixtures_dir, bad_id):
    doc = load_document(loader_fixtures_dir / "sample.txt").model_copy(update={"doc_id": bad_id})
    with pytest.raises(IngestionError, match="escapes"):
        save_document(doc, tmp_path / "documents")


def test_load_saved_document_missing_raises(tmp_path):
    with pytest.raises(IngestionError, match="no saved document"):
        load_saved_document(tmp_path, "nope")


# ---------- real corpus smoke ----------


def test_real_corpus_loads_cleanly():
    root = Path(__file__).parents[3] / "corpus" / "fastapi_docs"
    if not root.exists():
        pytest.skip("corpus not present")
    docs = [load_document(p, root=root) for p in iter_corpus_files(root)]
    assert len(docs) >= 100
    assert not any(d.doc_id.startswith("reference/") for d in docs)
    for d in docs:
        assert d.metadata["title"], d.doc_id
        for noise in ("{ #", "\n///", "<font", "<div class"):
            assert noise not in d.text, (d.doc_id, noise)
