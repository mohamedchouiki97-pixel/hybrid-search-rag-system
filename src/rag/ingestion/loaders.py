"""Loaders: turn a file into a Document with clean "light markdown" text.

Every format is normalized to the same shape: headings as `#` lines, code in
``` fences, prose as plain paragraphs. That lets one set of chunkers handle all
formats. PDFs also record where each page starts (metadata["page_offsets"]) so
chunkers can assign page numbers.
"""

from __future__ import annotations

import html
import io
import json
import re
import shutil
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader

from rag.core.interfaces import Loader
from rag.core.models import DocFormat, Document


class IngestionError(ValueError):
    """A file could not be turned into a Document. The message says why."""


SUPPORTED_EXTENSIONS: dict[str, DocFormat] = {
    ".md": DocFormat.MD,
    ".markdown": DocFormat.MD,
    ".txt": DocFormat.TXT,
    ".html": DocFormat.HTML,
    ".htm": DocFormat.HTML,
    ".pdf": DocFormat.PDF,
}

DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = ("reference",)

# ---------- shared text helpers ----------

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_WHITESPACE_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+-]*)")
_SHELL_LANGS = {"console", "bash", "shell", "sh", "powershell"}
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_INLINE_CODE_RE = re.compile(r"(`[^`]*`)")
_ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")
_ADMONITION_RE = re.compile(r"^\s*/{3,}\s*(\w+)\s*(?:\|\s*(.*?))?\s*$")  # "/// tip", "//// tab | Python 3.10+"
_ADMONITION_END_RE = re.compile(r"^\s*/{3,}\s*$")


def normalize_text(text: str) -> str:
    """Unicode NFC, \\n line endings, no trailing spaces, at most one blank line in a row."""
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def make_doc_id(path: str | Path, root: str | Path | None = None) -> str:
    """Relative path without extension ("tutorial/first-steps"); just the stem outside root."""
    p = Path(path).resolve()
    rel = Path(p.name)
    if root is not None:
        try:
            rel = p.relative_to(Path(root).resolve())
        except ValueError:
            pass
    return rel.with_suffix("").as_posix()


def _source_path(path: Path, root: str | Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _is_fence(line: str) -> str | None:
    m = _FENCE_RE.match(line)
    return m.group(1) if m else None


def _fence_lang(line: str) -> str:
    m = _FENCE_RE.match(line)
    return m.group(2).lower() if m else ""


def _closes_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(marker) and set(stripped) == {marker[0]}


def _strip_tags_outside_inline_code(line: str) -> str:
    parts = _INLINE_CODE_RE.split(line)
    for i in range(0, len(parts), 2):  # even parts are outside `inline code`
        parts[i] = html.unescape(_TAG_RE.sub("", parts[i]))
    return "".join(parts)


# ---------- markdown cleaning ----------


def clean_markdown(text: str) -> str:
    """Remove docs-site noise while keeping headings, prose and code.

    Outside code fences: drops `{ #anchor }` heading ids, turns `/// tip` admonitions
    into "Tip:", removes mkdocstrings `::: x.y` blocks, strips HTML tags (keeping
    their text). Inside fences: tags are stripped only from shell/console fences
    (terminal demos are coloured with <font>/<u> tags); every other fence,
    including `# comments` and HTML inside Python strings, is kept verbatim.
    """
    out: list[str] = []
    fence: str | None = None
    fence_is_shell = False
    skipping_directive = False
    for line in normalize_text(text).split("\n"):
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
                out.append(line)
            elif fence_is_shell:
                out.append(html.unescape(_TAG_RE.sub("", line)))
            else:
                out.append(line)
            continue

        if skipping_directive:
            if line == "" or line[0].isspace():
                continue
            skipping_directive = False

        if marker := _is_fence(line):
            fence = marker
            fence_is_shell = _fence_lang(line) in _SHELL_LANGS
            out.append(line)
            continue
        if line.startswith("::: "):
            skipping_directive = True
            continue
        if _ADMONITION_END_RE.match(line):
            continue
        if m := _ADMONITION_RE.match(line):
            kind, title = m.group(1).capitalize(), m.group(2)
            out.append(f"{kind}: {title}" if title else f"{kind}:")
            continue

        line = _ANCHOR_RE.sub("", line)
        out.append(_strip_tags_outside_inline_code(line))
    return normalize_text("\n".join(out))


def first_heading(text: str) -> str | None:
    """Text of the first `# ` heading outside code fences."""
    fence: str | None = None
    for line in text.split("\n"):
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
        elif marker := _is_fence(line):
            fence = marker
        elif line.startswith("# "):
            return line[2:].strip()
    return None


# ---------- loaders ----------


class BaseLoader:
    format: ClassVar[DocFormat]

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = root

    def load(self, path: str | Path) -> Document:
        p = Path(path)
        if not p.is_file():
            raise IngestionError(f"file not found: {p}")
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise IngestionError(f"could not read {p}: {exc}") from exc
        if not data.strip():
            raise IngestionError(f"empty file: {p}")
        text, metadata = self._parse(data, p)
        if not text:
            raise IngestionError(f"no text content after cleaning: {p}")
        return Document(
            doc_id=make_doc_id(p, self.root),
            source_path=_source_path(p, self.root),
            format=self.format,
            raw_path=str(p),
            text=text,
            metadata=metadata,
        )

    def _parse(self, data: bytes, path: Path) -> tuple[str, dict[str, Any]]:
        """Return (normalized text, metadata)."""
        raise NotImplementedError


class MarkdownLoader(BaseLoader):
    format = DocFormat.MD

    def _parse(self, data: bytes, path: Path) -> tuple[str, dict[str, Any]]:
        text = clean_markdown(decode_bytes(data))
        return text, {"title": first_heading(text)}


class TextLoader(BaseLoader):
    format = DocFormat.TXT

    def _parse(self, data: bytes, path: Path) -> tuple[str, dict[str, Any]]:
        return normalize_text(decode_bytes(data)), {}


class HtmlLoader(BaseLoader):
    format = DocFormat.HTML

    _DROP = ["script", "style", "nav", "header", "footer", "noscript", "template"]
    _BLOCKS = ["p", "div", "section", "article", "table", "tr", "ul", "ol", "blockquote"]

    def _parse(self, data: bytes, path: Path) -> tuple[str, dict[str, Any]]:
        soup = BeautifulSoup(decode_bytes(data), "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else None
        if title_tag:
            title_tag.decompose()
        for tag in soup.find_all(self._DROP):
            tag.decompose()
        # Like a browser: whitespace runs in text collapse to one space, except in <pre>.
        for s in soup.find_all(string=True):
            if type(s) is NavigableString and s.find_parent("pre") is None:
                s.replace_with(NavigableString(_WHITESPACE_RE.sub(" ", s)))

        for pre in soup.find_all("pre"):
            pre.replace_with(NavigableString(f"\n\n```\n{pre.get_text().strip(chr(10))}\n```\n\n"))
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                h.replace_with(NavigableString(f"\n\n{'#' * level} {h.get_text(' ', strip=True)}\n\n"))
        for li in soup.find_all("li"):
            li.insert_before(NavigableString("\n- "))
        for br in soup.find_all("br"):
            br.replace_with(NavigableString("\n"))
        for tag in soup.find_all(self._BLOCKS):
            tag.insert_before(NavigableString("\n\n"))
            tag.insert_after(NavigableString("\n\n"))

        text = self._collapse_prose_whitespace(soup.get_text())
        return text, {"title": title or first_heading(text)}

    @staticmethod
    def _collapse_prose_whitespace(text: str) -> str:
        out: list[str] = []
        fence: str | None = None
        for line in normalize_text(text).split("\n"):
            if fence is not None:
                if _closes_fence(line, fence):
                    fence = None
                out.append(line)
            elif marker := _is_fence(line):
                fence = marker
                out.append(line.strip())
            else:
                out.append(" ".join(line.split()))
        return normalize_text("\n".join(out))


class PdfLoader(BaseLoader):
    format = DocFormat.PDF

    def _parse(self, data: bytes, path: Path) -> tuple[str, dict[str, Any]]:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError(f"PDF is password protected: {path}")
            page_texts = [normalize_text(page.extract_text() or "") for page in reader.pages]
        except IngestionError:
            raise
        except Exception as exc:  # pypdf raises many types for malformed files
            raise IngestionError(f"could not read PDF {path}: {exc}") from exc

        parts: list[str] = []
        page_offsets: list[list[int]] = []  # [char offset where page starts, page number]
        offset = 0
        for number, page_text in enumerate(page_texts, start=1):
            if not page_text:
                continue
            if parts:
                offset += 2  # the "\n\n" joiner
            page_offsets.append([offset, number])
            parts.append(page_text)
            offset += len(page_text)
        if not parts:
            raise IngestionError(f"PDF has no extractable text (scanned images?): {path}")
        text = "\n\n".join(parts)
        return text, {"title": None, "page_count": len(page_texts), "page_offsets": page_offsets}


LOADERS: dict[DocFormat, type[BaseLoader]] = {
    DocFormat.MD: MarkdownLoader,
    DocFormat.TXT: TextLoader,
    DocFormat.HTML: HtmlLoader,
    DocFormat.PDF: PdfLoader,
}


def detect_format(path: str | Path) -> DocFormat:
    suffix = Path(path).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[suffix]
    except KeyError:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(f"unsupported file type {suffix or '(none)'!r}; supported: {supported}") from None


def get_loader(path: str | Path, root: str | Path | None = None) -> Loader:
    return LOADERS[detect_format(path)](root=root)


def load_document(path: str | Path, root: str | Path | None = None) -> Document:
    return get_loader(path, root).load(path)


def iter_corpus_files(root: str | Path, exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS) -> Iterator[Path]:
    """Supported files under root, sorted, skipping any path through an excluded folder."""
    root = Path(root)
    excluded = set(exclude_dirs)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if excluded.isdisjoint(p.relative_to(root).parts[:-1]):
                yield p


# ---------- raw + processed storage ----------


def _doc_dir(documents_dir: str | Path, doc_id: str) -> Path:
    base = Path(documents_dir).resolve()
    target = (base / doc_id).resolve()
    if target == base or base not in target.parents:
        raise IngestionError(f"doc_id {doc_id!r} escapes the documents folder")
    return target


def save_document(doc: Document, documents_dir: str | Path) -> Document:
    """Store the raw file and processed text under documents_dir/<doc_id>/.

    Returns the Document with raw_path pointing at the stored copy, so re-indexing
    never needs the original upload.
    """
    target = _doc_dir(documents_dir, doc.doc_id)
    target.mkdir(parents=True, exist_ok=True)
    src = Path(doc.raw_path)
    raw_dest = target / f"raw{src.suffix.lower()}"
    if src.resolve() != raw_dest.resolve():
        shutil.copyfile(src, raw_dest)
    saved = doc.model_copy(update={"raw_path": str(raw_dest)})
    (target / "text.md").write_text(saved.text, encoding="utf-8")
    (target / "document.json").write_text(saved.model_dump_json(exclude={"text"}, indent=2), encoding="utf-8")
    return saved


def load_saved_document(documents_dir: str | Path, doc_id: str) -> Document:
    """Rebuild a Document saved by save_document, without re-parsing the raw file."""
    target = _doc_dir(documents_dir, doc_id)
    try:
        meta = (target / "document.json").read_text(encoding="utf-8")
        text = (target / "text.md").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise IngestionError(f"no saved document for doc_id {doc_id!r}") from exc
    return Document.model_validate({**json.loads(meta), "text": text})
