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

### 2026-09-29: Citation.marker accepts 0
- Module: generation
- Change: `Citation.marker` is now `>= 0` (was `>= 1`).
- Why: an LLM can write `[0]`. DESIGN 5.4 says a marker with no matching chunk must be flagged, but `>= 1` made parsing crash instead. It is stored with `chunk_id=None, verified=False`.

### 2026-09-28: cache_path config key
- Module: indexing
- Change: `Settings.cache_path`, default `data/cache`.
- Why: DESIGN 10 says to cache embeddings by text hash, and evaluation will cache judge calls. Both need one configured folder.

### 2026-09-28: llm_provider config key
- Module: generation (planned)
- Change: `Settings.llm_provider` (`LLMProvider` enum: `openai` | `anthropic`), default `openai`. DESIGN 2 now names an OpenAI chat model for generation and the judge.
- Why: no Anthropic credit available. The provider is config, so the pipeline code only sees `LLMClient` and switching back needs no code changes.

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
