"""Chunkers: split a Document into Chunks. Three strategies, selectable by name.

- fixed:     sliding character window with overlap (baseline)
- recursive: one chunk per markdown section, oversized sections split at the most
             natural boundary; each chunk is prefixed with its heading breadcrumb
- semantic:  cut where embedding similarity between neighbouring sentences drops

Every chunker records section_heading and page_number from the chunk's start
offset in the document, and builds stable ids with make_chunk_id.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from rag.core.config import Settings
from rag.core.fakes import cosine
from rag.core.interfaces import Chunker, Embedder
from rag.core.models import Chunk, ChunkStrategy, Document, make_chunk_id
from rag.ingestion.loaders import closes_fence, fence_marker

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[*_])")
BREADCRUMB_SEP = " > "


# ---------- document structure ----------


@dataclass(frozen=True)
class Section:
    start: int  # offset of the heading line (0 for text before the first heading)
    body_start: int  # offset just after the heading line
    end: int
    breadcrumb: tuple[str, ...]

    @property
    def heading(self) -> str | None:
        return BREADCRUMB_SEP.join(self.breadcrumb) or None


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out, offset = [], 0
    for line in text.split("\n"):
        out.append((offset, line))
        offset += len(line) + 1
    return out


def parse_sections(text: str, max_level: int = 3) -> list[Section]:
    """Split at `#`..`###` headings outside code fences. Deeper headings stay in the body."""
    sections: list[Section] = []
    stack: list[str | None] = [None] * 6
    start, body_start, crumb = 0, 0, ()
    fence: str | None = None
    for offset, line in _lines_with_offsets(text):
        if fence is not None:
            if closes_fence(line, fence):
                fence = None
            continue
        if marker := fence_marker(line):
            fence = marker
            continue
        m = _HEADING_RE.match(line)
        if m is None or len(m.group(1)) > max_level:
            continue
        sections.append(Section(start, body_start, offset, crumb))
        level = len(m.group(1))
        stack[level - 1] = m.group(2)
        stack[level:] = [None] * (6 - level)
        crumb = tuple(t for t in stack[:level] if t)
        start, body_start = offset, min(offset + len(line) + 1, len(text))
    sections.append(Section(start, body_start, len(text), crumb))
    return sections


class _Locator:
    """Maps a character offset to its section heading and PDF page number."""

    def __init__(self, doc: Document) -> None:
        self.sections = parse_sections(doc.text)
        self._section_starts = [s.start for s in self.sections]
        pages = doc.metadata.get("page_offsets") or []
        self._page_starts = [int(o) for o, _ in pages]
        self._page_numbers = [int(n) for _, n in pages]

    def heading_at(self, offset: int) -> str | None:
        i = bisect.bisect_right(self._section_starts, offset) - 1
        return self.sections[max(i, 0)].heading

    def page_at(self, offset: int) -> int | None:
        if not self._page_starts:
            return None
        i = bisect.bisect_right(self._page_starts, offset) - 1
        return self._page_numbers[max(i, 0)]


@dataclass(frozen=True)
class _Piece:
    text: str
    start: int  # offset in doc.text, for page lookup
    heading: str | None


def _to_chunks(doc: Document, strategy: ChunkStrategy, pieces: Sequence[_Piece], locator: _Locator) -> list[Chunk]:
    chunks: list[Chunk] = []
    for piece in pieces:
        if not piece.text.strip():
            continue
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, strategy, index),
                doc_id=doc.doc_id,
                text=piece.text,
                source_path=doc.source_path,
                section_heading=piece.heading,
                page_number=locator.page_at(piece.start),
                chunk_index=index,
                strategy=strategy,
            )
        )
    return chunks


def _split_long(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Split at paragraph, then line, then word boundaries. Returns (text, start offset in `text`)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True, separators=["\n\n", "\n", " ", ""]
    )
    return [(d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])]


# ---------- strategies ----------


class FixedChunker:
    """Baseline: every chunk_size characters, sharing chunk_overlap characters with its neighbour."""

    strategy = ChunkStrategy.FIXED

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("need 0 <= chunk_overlap < chunk_size")
        self._splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            strip_whitespace=False, add_start_index=True,
        )  # fmt: skip

    def chunk(self, doc: Document) -> list[Chunk]:
        locator = _Locator(doc)
        pieces = [
            _Piece(d.page_content, d.metadata["start_index"], locator.heading_at(d.metadata["start_index"]))
            for d in self._splitter.create_documents([doc.text])
        ]
        return _to_chunks(doc, self.strategy, pieces, locator)


class RecursiveChunker:
    """One chunk per section when it fits; otherwise split inside the section only.

    Each chunk starts with its heading breadcrumb ("First Steps > Run it") so the
    chunk says what it is about even out of context. The breadcrumb counts toward
    chunk_size.
    """

    strategy = ChunkStrategy.RECURSIVE

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("need 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        locator = _Locator(doc)
        pieces: list[_Piece] = []
        for section in locator.sections:
            raw = doc.text[section.body_start : section.end]
            body = raw.lstrip("\n").rstrip()
            if not body:
                continue
            body_offset = section.body_start + (len(raw) - len(raw.lstrip("\n")))
            prefix = f"{section.heading}\n\n" if section.heading else ""
            if len(prefix) + len(body) <= self.chunk_size:
                pieces.append(_Piece(prefix + body, body_offset, section.heading))
                continue
            budget = max(self.chunk_size - len(prefix), self.chunk_size // 2)
            overlap = min(self.chunk_overlap, budget // 2)
            for text, start in _split_long(body, budget, overlap):
                pieces.append(_Piece(prefix + text, body_offset + start, section.heading))
        return _to_chunks(doc, self.strategy, pieces, locator)


def split_units(text: str) -> list[tuple[int, int]]:
    """Sentence-level (start, end) spans for semantic chunking.

    Code fences are one unit each (never split). A heading is glued to the unit
    after it so it is never a chunk on its own.
    """
    blocks: list[tuple[int, int, str]] = []
    para: list[int] | None = None  # [start, end]
    fence: str | None = None
    code_start = 0

    def close_para() -> None:
        nonlocal para
        if para is not None:
            blocks.append((para[0], para[1], "para"))
            para = None

    for offset, line in _lines_with_offsets(text):
        end = offset + len(line)
        if fence is not None:
            if closes_fence(line, fence):
                blocks.append((code_start, end, "code"))
                fence = None
            continue
        if marker := fence_marker(line):
            close_para()
            fence, code_start = marker, offset
        elif not line.strip():
            close_para()
        elif _HEADING_RE.match(line):
            close_para()
            blocks.append((offset, end, "heading"))
        elif para is None:
            para = [offset, end]
        else:
            para[1] = end
    if fence is not None:
        blocks.append((code_start, len(text), "code"))
    close_para()

    units: list[tuple[int, int]] = []
    pending_heading: int | None = None
    for start, end, kind in blocks:
        if kind == "heading":
            if pending_heading is None:
                pending_heading = start
            continue
        spans = [(start, end)]
        if kind == "para":
            spans, s = [], start
            for m in _SENTENCE_BREAK_RE.finditer(text, start, end):
                spans.append((s, m.start()))
                s = m.end()
            spans.append((s, end))
        if pending_heading is not None:
            spans[0] = (pending_heading, spans[0][1])
            pending_heading = None
        units.extend(spans)
    if pending_heading is not None:
        units.append((pending_heading, len(text.rstrip())))
    return units


def breakpoints(similarities: Sequence[float], percentile: float) -> set[int]:
    """Indexes i where the cut goes between unit i and i+1: similarity in the bottom `percentile`."""
    if not similarities:
        return set()
    ordered = sorted(similarities)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    threshold = ordered[rank]
    return {i for i, s in enumerate(similarities) if s <= threshold}


class SemanticChunker:
    """Cut where the topic changes, judged by embeddings.

    1. Split into sentence units (code blocks whole).
    2. Embed every unit; cosine similarity between each pair of neighbours.
    3. Cut where similarity is in this document's bottom `breakpoint_percentile`
       (relative, so it adapts to any embedding model).
    4. Enforce size: split groups over chunk_size, merge groups under min_chunk_chars.
    """

    strategy = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 800,
        breakpoint_percentile: float = 0.10,
        min_chunk_chars: int | None = None,
    ) -> None:
        if not 0 < breakpoint_percentile < 1:
            raise ValueError("breakpoint_percentile must be between 0 and 1")
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_chars = chunk_size // 8 if min_chunk_chars is None else min_chunk_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        units = split_units(text)
        if not units:
            return []
        vectors = self.embedder.embed([text[s:e] for s, e in units])
        sims = [cosine(vectors[i], vectors[i + 1]) for i in range(len(units) - 1)]
        cuts = breakpoints(sims, self.breakpoint_percentile)

        groups: list[list[tuple[int, int]]] = [[units[0]]]
        for i, unit in enumerate(units[1:]):
            if i in cuts:
                groups.append([])
            groups[-1].append(unit)

        spans: list[tuple[int, int]] = []
        for group in groups:
            spans.extend(self._fit(text, group))
        spans = self._merge_small(spans)

        locator = _Locator(doc)
        pieces = [_Piece(text[s:e], s, locator.heading_at(s)) for s, e in spans]
        return _to_chunks(doc, self.strategy, pieces, locator)

    def _fit(self, text: str, group: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Split a group into spans of at most chunk_size, keeping units whole where possible."""
        spans: list[tuple[int, int]] = []
        run: tuple[int, int] | None = None
        for start, end in group:
            if end - start > self.chunk_size:  # one unit too big (e.g. a long code block)
                if run:
                    spans.append(run)
                    run = None
                spans.extend(
                    (start + off, start + off + len(t)) for t, off in _split_long(text[start:end], self.chunk_size, 0)
                )
            elif run is None:
                run = (start, end)
            elif end - run[0] <= self.chunk_size:
                run = (run[0], end)
            else:
                spans.append(run)
                run = (start, end)
        if run:
            spans.append(run)
        return spans

    def _merge_small(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and (end - start < self.min_chunk_chars or merged[-1][1] - merged[-1][0] < self.min_chunk_chars):
                if end - merged[-1][0] <= self.chunk_size:
                    merged[-1] = (merged[-1][0], end)
                    continue
            merged.append((start, end))
        return merged


# ---------- factory ----------


def make_chunker(
    strategy: ChunkStrategy | str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embedder: Embedder | None = None,
) -> Chunker:
    strategy = ChunkStrategy(strategy)
    if strategy is ChunkStrategy.FIXED:
        return FixedChunker(chunk_size, chunk_overlap)
    if strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size, chunk_overlap)
    if embedder is None:
        raise ValueError("the semantic chunker needs an embedder")
    return SemanticChunker(embedder, chunk_size)


def get_chunker(settings: Settings, embedder: Embedder | None = None) -> Chunker:
    """The chunker named by settings.chunk_strategy."""
    return make_chunker(settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap, embedder)
