> **Historical document.** This is the original build plan the project was developed from, kept as written, with later decisions folded in (OpenAI as the default LLM, core "stable, not frozen"). It was written as instructions for the build sessions, hence the imperative tone. For how the system works today see the [README](../README.md), and for results see the [case study](CASE_STUDY.md).

# Hybrid Search RAG System: Design and Build Plan

Read this file fully before writing any code. You will be told which modules to build in your session. Only build those.

## 1. What We Are Building

A system that answers questions about a set of documents using only those documents, with checkable citations.

Question flow:
1. Ingest: load docs, clean them, split into chunks.
2. Index: store each chunk in a vector DB (meaning search) and a BM25 index (keyword search). Skip near-duplicates.
3. Retrieve: search both, merge with Reciprocal Rank Fusion (RRF), rerank, keep top 5.
4. Answer: LLM answers only from those chunks, with [1], [2] style citations.
5. Verify: a judge checks each citation supports its claim. Score confidence. If confidence is low, return a structured "I don't know" instead of guessing.
6. Evaluate: a 50+ question golden set measures correctness, faithfulness, retrieval relevance, citation accuracy. Compare three chunking strategies.

Shipped as: FastAPI service, Streamlit dashboard, Docker compose, eval report, case study.

## 2. Decisions (fixed, do not change without asking)

- Language: Python 3.11+
- Embeddings: OpenAI text-embedding-3-small
- Generation and judge: an OpenAI chat model (name read from the LLM_MODEL env var, never hardcoded). The provider is config (llm_provider: openai | anthropic, default openai); the Anthropic client is optional.
- Vector store: ChromaDB (persistent, file based)
- Sparse search: rank_bm25
- Reranker: local cross-encoder, cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers
- Chunking: LangChain text splitters
- API: FastAPI with Pydantic v2
- Dashboard: Streamlit
- Tests: pytest
- Packaging: single pyproject.toml
- Corpus: a public markdown docs set (default: the FastAPI docs, MIT licensed, 50 to 100 pages). Corpus path is config, so it can be swapped.

## 3. Repo Layout

```
rag/
  pyproject.toml
  docker-compose.yml
  .env.example
  docs/DESIGN.md
  docs/CORE_CHANGELOG.md    (changelog: every core change and why)
  src/rag/
    core/                   shared models, interfaces, config, fakes (stable after Step 0)
      models.py
      interfaces.py
      config.py
      fakes.py
    ingestion/
      loaders.py
      chunkers.py
    indexing/
      embedder.py
      vector_store.py
      sparse_index.py
      indexer.py            embed, store, dedup, keep both indexes in sync
    retrieval/
      dense.py
      sparse.py
      fusion.py
      reranker.py
      retriever.py
    generation/
      llm_client.py
      prompts.py
      generator.py
      citation_verifier.py
      confidence.py
      abstain.py
    evaluation/
      golden/qa.jsonl
      metrics.py
      runner.py
      chunking_report.py
    api/
      main.py
      routes.py
      service.py            wires all modules together
    dashboard/
      app.py
  tests/
    fixtures/               small sample corpus, 5 docs
    unit/<module>/
    integration/
    live/                   tests that call real APIs, run by hand
  scripts/
    seed.py
```

## 4. Shared Contracts (built first, in Step 0)

Everything below lives in src/rag/core. Every module codes against these and only these. After Step 0, core is stable. Core may be edited directly when a module needs it, as long as all existing tests still pass. Record each change and why in docs/CORE_CHANGELOG.md. Modules still talk to each other only through core's interfaces and fakes.

### 4.1 Data models (models.py)

- Document: doc_id, source_path, format (md, txt, html, pdf), raw_path, text, metadata dict
- Chunk: chunk_id, doc_id, text, source_path, section_heading (optional), page_number (optional), chunk_index, strategy (fixed, recursive, semantic), char_count
- RetrievedChunk: chunk, score, dense_rank (optional), sparse_rank (optional), rerank_score (optional)
- Citation: marker (int), chunk_id, claim_text, verified (bool or None), judge_reason
- Confidence: retrieval, citation_coverage, completeness, composite (all floats 0 to 1)
- Answer: question, answer_text, citations list, confidence, abstained (bool), found (str), missing (str), suggested_docs (list of doc_id), retrieved (list of RetrievedChunk)
- AskRequest: question, mode (hybrid or dense), top_k (default 5)
- IngestResult: doc_id, chunks_added, chunks_skipped_duplicate

### 4.2 Interfaces (interfaces.py, typing.Protocol)

- Loader.load(path) returns Document
- Chunker.chunk(doc) returns list of Chunk
- Embedder.embed(texts) returns list of vectors
- VectorStore: add(chunks, vectors), query(vector, k) returns list of (Chunk, score), count(), delete_doc(doc_id)
- SparseIndex: add(chunks), query(text, k) returns list of (Chunk, score), delete_doc(doc_id)
- Retriever.retrieve(question, mode, k) returns list of RetrievedChunk (already fused and reranked in hybrid mode)
- Reranker.rerank(question, chunks, top_n) returns list of RetrievedChunk
- LLMClient.complete(system, user) returns str
- Generator.answer(question, chunks) returns Answer (citations parsed, not yet verified)
- CitationVerifier.verify(answer, chunks) returns Answer (verified flags filled in)

### 4.3 Fakes (fakes.py)

These let any module be tested without the others and without paying for API calls.

- FakeEmbedder: deterministic vectors from a hash of the text. Same text always gives the same vector. Similar text does not need to be close.
- FakeLLM: returns canned strings keyed by a substring of the prompt, with a default fallback.
- FakeReranker: sorts by simple word overlap with the question.
- Fixture corpus: 5 short markdown docs with known facts, used by every integration test.

### 4.4 Config keys (config.py)

Add all of these up front: chunk_strategy, chunk_size, chunk_overlap, dedup_threshold, dense_k, sparse_k, rrf_dense_weight, rrf_sparse_weight, rerank_candidates, rerank_top_n, abstain_threshold, confidence_weights, llm_model, embedding_model, corpus_path, chroma_path.

## 5. Module Specs

Every module writes its own unit tests (section 6) as it goes. Run pytest -m unit before each commit.

### 5.1 Ingestion

Loaders: markdown, text, HTML, PDF. Output plain text plus metadata (source file, section heading, page number). Save the raw file next to the processed text so re-indexing does not need a re-upload.

Chunkers (all implement Chunker, selectable by name from config):
1. Fixed size with overlap (baseline).
2. Recursive split by section headers (structure aware).
3. Semantic: split where embedding similarity between neighbouring sentences drops. Takes an Embedder by injection.

Every chunk records which strategy made it. Chunk ids must be stable (hash of doc_id, strategy, index) so re-ingesting gives the same ids.

Unit tests:
- each loader returns expected text and metadata for a fixture file
- broken or empty file raises a clear error
- fixed chunker respects size and overlap
- recursive chunker never splits in the middle of a header section when it fits
- semantic chunker splits at a planted topic change (FakeEmbedder with hand set vectors)
- chunk ids are stable across runs
- strategy field is set correctly

### 5.2 Indexing

Indexer: for a list of chunks, embed, check for near-duplicates (cosine above 0.95 against existing chunks, threshold in config), skip and count duplicates, add the rest to ChromaDB and BM25 together. Both indexes must always hold the same chunk ids. If one write fails, roll back the other.

Unit tests:
- dedup skips a chunk with cosine above the threshold and keeps one below it
- indexer keeps both indexes in sync, including after a simulated failure
- delete_doc removes from both indexes

### 5.3 Retrieval

- dense: top k by cosine (default k=10)
- sparse: BM25 top k
- fusion: RRF with configurable weights (default 0.7 dense, 0.3 sparse)
- reranker: take top 20 from fusion, score with the cross-encoder, keep top 5
- retriever: ties it together, supports mode "dense" (dense only, for the comparison toggle) and "hybrid"

Unit tests:
- RRF math on hand written ranked lists (known expected order)
- RRF weights change the order as expected
- an item in both lists beats an item in one list
- BM25 finds an exact rare token (like an error code) that dense misses (use fake vectors that do not match)
- reranker returns top_n sorted by score

### 5.4 Generation and Citations

- prompts.py: system prompt says answer only from context, cite with [n], say so when context is insufficient. Context is numbered blocks.
- generator: builds the prompt, calls LLMClient, parses [n] markers into Citation objects with the claim sentence they are attached to.
- citation_verifier: for each citation, send claim and cited chunk to the judge, get supported yes or no plus a reason. Run calls concurrently. Flag unsupported.
- confidence: retrieval confidence (from top chunk scores), citation coverage (share of claims with verified citations), completeness (judge call: did it address every part of the question). Composite is a weighted mean, weights in config.
- abstain: if retrieval confidence is below threshold, skip generation and return an Answer with abstained true, found, missing, and suggested_docs filled in.

Unit tests (all with FakeLLM):
- prompt contains every chunk, numbered correctly
- parser handles [1], [1][2], [1, 2], and no citations
- parser ignores a marker with no matching chunk and flags it
- verifier marks a citation unsupported when the judge says no
- coverage math on hand built examples
- composite confidence is within 0 to 1 and matches the weights
- abstain path triggers below threshold and never calls the generation LLM
- malformed judge output is handled without crashing

### 5.5 Evaluation

- golden/qa.jsonl: 50+ hand written questions tied to the corpus. Fields: id, question, gold_answer, gold_chunk_sections, type. Types: lookup, multi_hop, no_answer, ambiguous. Target mix: 25 lookup, 10 multi hop, 8 no answer, 7 ambiguous.
- metrics: answer correctness (judge vs gold), faithfulness (are all claims grounded in retrieved context), retrieval relevance (were gold sections retrieved, recall at k and MRR), citation accuracy (from verifier flags). For no_answer questions, correct means the system abstained.
- runner: takes a pipeline object, runs all questions, writes a JSON results file and a markdown summary. Caches judge calls on disk by hash so re-runs are cheap.
- chunking_report: runs the suite once per chunking strategy and writes a comparison table.

Unit tests:
- each metric on hand built examples with known scores
- recall at k and MRR on known rankings
- no_answer scoring rewards abstaining and penalises answering
- runner works end to end against a stub pipeline (returns canned Answers)
- judge cache returns the stored result and skips the LLM
- golden file validator: every line parses, ids unique, at least 50 rows, all four types present

### 5.6 API, Dashboard, Docker

API (FastAPI):
- POST /v1/ask takes AskRequest, returns Answer (answer, citations, confidence, sources)
- GET /v1/documents lists indexed docs with chunk counts
- POST /v1/ingest takes a file upload, returns IngestResult
- GET /healthz
- OpenAPI docs with request and response examples

service.py builds the pipeline from config using the interfaces. It is the only place that imports from every module.

Dashboard (Streamlit): question box, answer with clickable citations that show the chunk, retrieved chunks ranked with scores, confidence broken down by dimension, and a hybrid vs dense-only side by side toggle.

Docker: docker-compose with api, chroma data volume, dashboard. scripts/seed.py ingests the sample corpus so reviewers can run it right away.

Unit tests:
- each route returns the right status and schema using FastAPI TestClient with the service built from fakes
- bad input returns 422, empty question rejected
- ingest of an unsupported file type returns a clear 4xx
- service builds with each chunking strategy from config
- dashboard helper functions (citation linking, score formatting) tested as plain functions

## 6. Testing Strategy

Three layers, marked with pytest markers.

1. unit (marker: unit). Fast, no network, no API keys. Uses fakes only. Runs on every commit. Target: 85 percent line coverage per module.
2. integration (marker: integration). Real ChromaDB in a temp dir, real BM25, FakeReranker or the real cross-encoder, FakeEmbedder and FakeLLM. No network. Written after the modules are merged. Runs before every merge to main.
3. live (marker: live). Calls the real OpenAI APIs (and Anthropic, if that provider is configured). A handful of tests, run by hand. Checks that the real embedder returns the right vector size and the real judge returns parseable output.

Commands:
```
pytest -m unit
pytest -m integration
pytest -m live
```

Integration tests to write:
- ingest to index: ingest the 5 fixture docs, confirm chunk counts, both indexes hold identical ids, re-ingesting adds zero (dedup works)
- index to retrieve: query with a known fact, expected chunk is in the top 5, hybrid finds an exact token that dense mode misses
- retrieve to generate: with FakeLLM, answer has citations that map to real retrieved chunks
- abstain path: question with no answer in fixtures returns abstained true and never reaches the generator
- verify path: a planted bad citation gets flagged
- API: upload a file through POST /v1/ingest, then POST /v1/ask, get a full Answer
- eval smoke: run the eval runner on 5 golden questions with fakes and confirm a report file is written
- chunking switch: same ingest run under all three strategies works and tags chunks correctly

Rules for all tests:
- no test may depend on another test's state
- no unit test may touch the network
- every bug fix gets a regression test

## 7. Working Rules

1. Build only the modules you were assigned, and only edit those folders and their tests.
2. Core is stable, not frozen: edit src/rag/core directly when a module needs it, keep all existing tests passing, and log the change and why in docs/CORE_CHANGELOG.md.
3. Code against the interfaces and fakes in core, not against other modules' real code.
4. Small commits with clear messages. Run pytest -m unit before each one.
5. When done, report: what you built, test results, any contract requests, and anything likely to break at integration.

## 8. Build Steps

Step 0: core, fixture corpus, pytest setup. Nothing else can start until this is done.
Step 1: Ingestion, Indexing, Retrieval, Generation, Evaluation (each depends only on core).
Step 2: merge, then API, Dashboard, Docker (needs the real modules).
Step 3: integration tests, live tests.
Step 4: run the full eval on real models, then tune thresholds and weights.
Step 5: chunking comparison report, README with architecture diagram, case study, demo.

## 9. Definition of Done

Per module:
- all listed unit tests pass
- coverage target met
- no network calls in unit tests

Whole project:
- docker compose up starts api, chroma, dashboard, and the seed script indexes the sample corpus
- pytest -m unit and pytest -m integration pass on a clean checkout
- eval report exists with correctness, faithfulness, retrieval relevance, citation accuracy on 50+ questions
- chunking strategy comparison table exists
- hybrid vs dense-only comparison exists
- README has architecture diagram, quickstart, and the headline numbers
- case study written, leading with the numbers

## 10. Risks

- Contract drift: mitigated by keeping core stable (edits only with all tests passing) and logging every change in docs/CORE_CHANGELOG.md.
- Judge cost: eval makes many LLM calls. Cache judge results on disk and run live evals rarely.
- Golden set quality: weak questions give weak numbers. Write them by hand from the actual corpus and review them.
- Semantic chunking is slow and costs embeddings. Cache embeddings by text hash.
- Cross-encoder first run downloads a model. Pin the model name and cache it in the Docker image.
