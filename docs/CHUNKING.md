# Chunking strategy comparison

Same 50 questions, same retrieval (hybrid), same model; only the chunking changes. Final run (`reports/eval/final/chunking_comparison.md`):

| Strategy | Chunks | Correctness | Faithfulness | Recall@5 | MRR | Citation acc. |
|---|---|---|---|---|---|---|
| fixed (800 chars, 100 overlap) | 1,382 | 0.92 | 0.99 | 0.74 | **0.58** | 0.96 |
| **recursive** (by headings) | 1,865 | **0.92** | 0.98 | **0.81** | **0.80** | 0.96 |
| semantic (topic shifts) | 1,488 | 0.87 | 1.00 | 0.72 | 0.66 | 0.97 |

Correctness differences under about 0.03 are run-to-run noise (see `reports/eval/RESULTS.md`).

## What each strategy does

- **Fixed** cuts every 800 characters, with 100 characters shared between neighbours so a fact on a boundary is not split in half. It is the control group.
- **Recursive** makes one chunk per Markdown section when it fits, splits longer sections at paragraph, then line, then word boundaries, and **prefixes every chunk with its heading breadcrumb** (for example `First Steps > Interactive API docs`).
- **Semantic** embeds every sentence and cuts where similarity between neighbouring sentences falls into the document's bottom 10%, then enforces size limits. Code blocks stay whole.

## Why recursive wins

**It ranks the right section first.** Recall@5 is only 0.07 apart from fixed, so both usually *find* the gold section. But MRR (how high it ranks) is 0.80 vs 0.58. A fixed window often starts mid-section and mixes two topics, so it matches a question only partially. A recursive chunk is one topic, and its breadcrumb tells both the embedding and BM25 what that topic is.

**Semantic loses heading context.** It finds topic boundaries well, but its chunks carry no breadcrumb. The FastAPI docs repeat code examples across pages, and without a heading those chunks become identical text. At indexing time 331 semantic chunks were dropped as near-duplicates: 259 of them byte-identical to a chunk already kept, most of the rest differing only by a filename comment. Recursive dropped 144, only 16 of them identical. The deduplication was right both times; semantic chunks simply had less to tell them apart.

## Things we checked along the way

- **Code blocks are the main trap in this corpus.** 486 lines inside code fences start with `# ` (Python comments). A naive header splitter treats them as headings. LangChain's `MarkdownHeaderTextSplitter` avoids that but strips code indentation (`    return 1` became `return 1`), so recursive chunking uses a small fence-aware splitter of our own and LangChain's `RecursiveCharacterTextSplitter` only inside oversized sections.
- **Chunk IDs are stable:** a hash of document, strategy and position. Re-ingesting a document replaces its chunks instead of duplicating them.

## If we kept going

Give semantic chunks the same breadcrumb prefix as recursive ones, then re-run. That would separate "semantic boundaries" from "missing headings" as the cause of the gap.
