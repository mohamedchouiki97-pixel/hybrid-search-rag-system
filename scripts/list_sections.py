"""Write the catalog of valid (doc_id, section) pairs for the golden dataset.

Every gold_chunk_sections entry in qa.jsonl must match an entry here exactly. The
sections come from the same loader and section parser the chunkers use, so a
section listed here is exactly the section_heading chunks will carry.

The catalog records a fingerprint of the corpus and of the code that produces
headings. If either changes, re-run this script and re-check the golden set.

    uv run python scripts/list_sections.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag.core.config import Settings
from rag.ingestion import chunkers, loaders
from rag.ingestion.chunkers import parse_sections
from rag.ingestion.loaders import iter_corpus_files, load_document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "rag" / "evaluation" / "golden" / "sections.json"


def fingerprint(corpus: Path, files: list[Path]) -> dict[str, str]:
    corpus_hash = hashlib.sha256()
    for p in files:
        corpus_hash.update(p.relative_to(corpus).as_posix().encode())
        corpus_hash.update(p.read_bytes())
    code_hash = hashlib.sha256()
    for module in (loaders, chunkers):
        code_hash.update(Path(module.__file__).read_bytes())
    return {"corpus_sha256": corpus_hash.hexdigest()[:16], "heading_code_sha256": code_hash.hexdigest()[:16]}


def build_catalog(corpus: Path) -> dict:
    files = list(iter_corpus_files(corpus))
    entries = []
    for path in files:
        doc = load_document(path, root=corpus)
        for section in parse_sections(doc.text):
            body = doc.text[section.body_start : section.end].strip()
            if section.heading and body:  # only sections that actually produce chunks
                entries.append({"doc_id": doc.doc_id, "section": section.heading})
    return {
        "corpus_path": corpus.relative_to(ROOT).as_posix() if corpus.is_relative_to(ROOT) else str(corpus),
        **fingerprint(corpus, files),
        "doc_count": len(files),
        "section_count": len(entries),
        "sections": entries,
    }


def main() -> None:
    settings = Settings(llm_model="unused")  # type: ignore[call-arg]
    corpus = (ROOT / settings.corpus_path).resolve()
    catalog = build_catalog(corpus)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{catalog['section_count']} sections in {catalog['doc_count']} docs -> {OUT.relative_to(ROOT)}")
    print(f"corpus {catalog['corpus_sha256']}  heading code {catalog['heading_code_sha256']}")


if __name__ == "__main__":
    main()
