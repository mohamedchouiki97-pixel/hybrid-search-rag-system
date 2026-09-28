# Core Changelog

`src/rag/core` is stable, not frozen. Edit it directly when a module needs a
change, as long as every existing test still passes. Log each change here,
newest first, with the reason. Modules still talk to each other only through
core's interfaces and fakes.

## Template

```
### YYYY-MM-DD: <short title>
- Module: <which module needed it>
- Change: <what changed in core>
- Why: <what was missing or awkward without it>
```

## Changes

### 2026-09-28: documents_path config key
- Module: ingestion
- Change: `Settings.documents_path`, default `data/documents`.
- Why: DESIGN 5.1 says to keep the raw file next to the processed text so re-indexing needs no re-upload. `save_document()` needs a configured folder for that.

### 2026-09-26: Step 0 additions beyond DESIGN 4
- Module: core (Step 0)
- Change:
  - `VectorStore` and `SparseIndex` gained `delete(chunk_ids)` and `ids()`.
  - `VectorStore` gained `list_docs()`, returning the new `DocumentSummary` model.
  - `Citation.chunk_id` is optional.
  - Config gained `rrf_k` and `reranker_model`.
  - `fakes.py` gained `InMemoryVectorStore` and `InMemorySparseIndex`.
  - `corpus_path` defaults to `corpus/fastapi_docs`.
- Why:
  - Rolling back a partial index write needs delete-by-id. `delete_doc` would also wipe the doc's older chunks.
  - `ids()` makes the "both indexes in sync" check simple.
  - `GET /v1/documents` needs chunk counts per document.
  - A `[n]` marker with no matching chunk must be representable (`chunk_id=None`).
  - The RRF constant and the reranker model name should not be hardcoded.
  - The in-memory fakes let retrieval and generation be tested without Chroma.
