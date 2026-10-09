"""Index the corpus so the API has something to answer from.

    uv run python scripts/seed.py                    # all three strategies
    uv run python scripts/seed.py --strategy recursive
    uv run python scripts/seed.py --force            # re-index even if already indexed

Needs OPENAI_API_KEY (embeddings) but no LLM. Embeddings are cached on disk, so
re-running is cheap. A strategy whose index already has chunks is skipped unless
--force is given, which makes this safe to run on every `docker compose up`.
"""

from __future__ import annotations

import argparse
import sys
import time

from rag.api.service import build_ingestion, for_strategy
from rag.core.config import Settings
from rag.core.models import ChunkStrategy
from rag.indexing.embedder import make_embedder
from rag.ingestion.loaders import IngestionError, iter_corpus_files, load_document, save_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["all", *(s.value for s in ChunkStrategy)], default="all")
    parser.add_argument("--force", action="store_true", help="re-index strategies that already have chunks")
    parser.add_argument("--limit", type=int, default=None, help="only the first N corpus files (for a quick try)")
    args = parser.parse_args()

    settings = Settings(llm_model="not-needed-for-seeding")  # type: ignore[call-arg]
    if settings.openai_api_key is None:
        print("OPENAI_API_KEY is not set (needed for embeddings).", file=sys.stderr)
        return 1
    files = list(iter_corpus_files(settings.corpus_path))[: args.limit]
    if not files:
        print(f"No supported files under {settings.corpus_path}.", file=sys.stderr)
        return 1

    embedder = make_embedder(settings)
    strategies = list(ChunkStrategy) if args.strategy == "all" else [ChunkStrategy(args.strategy)]
    for strategy in strategies:
        chunker, indexer = build_ingestion(for_strategy(settings, strategy), embedder)
        if indexer.vector_store.count() and not args.force:
            print(
                f"[{strategy}] already indexed ({indexer.vector_store.count()} chunks); skipping. Use --force to redo."
            )
            continue
        start, added, skipped = time.perf_counter(), 0, 0
        for n, path in enumerate(files, start=1):
            try:
                doc = save_document(load_document(path, root=settings.corpus_path), settings.documents_path)
            except IngestionError as exc:
                print(f"[{strategy}] skip {path}: {exc}")
                continue
            result = indexer.index_document(doc.doc_id, chunker.chunk(doc))
            added, skipped = added + result.chunks_added, skipped + result.chunks_skipped_duplicate
            print(f"\r[{strategy}] {n}/{len(files)} docs", end="", flush=True)
        print(
            f"\r[{strategy}] {len(files)} docs -> {added} chunks, {skipped} near-duplicates skipped, "
            f"indexes in sync: {indexer.in_sync()} ({time.perf_counter() - start:.0f}s)"
        )
    print(f"Embedding cache: {embedder.hits} hits, {embedder.misses} new embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
