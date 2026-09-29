# Case study: making a RAG system honest about what it knows

## The numbers

On 50 hand-written questions about the FastAPI documentation (recursive chunks, hybrid retrieval, `gpt-4.1-mini-2025-04-14`):

| | Baseline | Final |
|---|---|---|
| Answer correctness | 0.85 | **0.92** |
| Faithfulness | 0.93 | **0.98** |
| Citation accuracy | 0.88 | **0.96** |
| Unanswerable questions refused | 5 of 8 | **8 of 8** |
| Answerable questions wrongly refused | 0 | **0** |

The gains held in every configuration: fixed chunks (correctness 0.83 → 0.92), semantic chunks (0.77 → 0.87) and dense-only retrieval (0.74 → 0.89, with refusals going from 0 of 8 to 8 of 8). Retrieval metrics did not move, because none of the changes touched retrieval. That makes it a clean before/after.

## The problem

A documentation assistant that answers confidently from memory is worse than none: users can't tell a grounded answer from a plausible guess. The goals were:
1. answer only from the retrieved documentation;
2. cite every claim, and verify each citation with a second model;
3. report confidence per dimension;
4. say "I don't know" in a structured, useful way when the docs don't cover the question.

## The system

- **Hybrid retrieval:** dense search (OpenAI embeddings in Chroma) plus BM25, merged with Reciprocal Rank Fusion, then reranked by a local cross-encoder.
- **Answering:** the model answers from numbered passages and cites them.
- **Checking:** a judge verifies claims against the passages they cite.
- **Confidence** combines retrieval strength, citation coverage and completeness.
- **Built in isolation:** each module was built and tested against shared interfaces and fakes before anything was wired together. The result is 381 unit tests with no network, 12 integration tests on real indexes, and 3 live tests.

## What real data exposed

The first end-to-end run looked fine question by question. Measurement across the 50 questions told a different story. Each fix below was measured, not assumed.

**1. Correct answers were scored as poorly cited.** The model often cites once at the end of a paragraph: "A. B. C [1][2]." The coverage metric counted A and B as uncited, so a perfect answer scored 0.33.
*Fix:* uncited sentences are grouped with the next cited sentence in the same paragraph, and the judge checks the whole group. Nothing is trusted unchecked. On that question coverage went 0.33 → 1.00.

**2. Facts combined from two passages were marked unsupported.** The judge checked each citation alone, so a sentence built from [1] and [3] failed on both.
*Fix:* one judge call per claim, with all of its cited passages. The judge also says which passages it needed, so an irrelevant citation is still caught. Citation accuracy 0.88 → 0.96, with fewer judge calls.

**3. Confidence meant different things in different modes.** Hybrid confidence was a reranker probability (0.99), dense confidence was a cosine similarity (0.60) for the same question. So one threshold could not fit both, and dense mode refused **0 of 8** unanswerable questions.
*Fix:* dense mode keeps its ranking but is scored by the same cross-encoder. Implementing it exposed a bug of our own: confidence used the *first* chunk, but in dense mode the best chunk can be third. It now uses the best chunk.

**4. Correct refusals were counted as failures.** Three "failed" unanswerable questions were not hallucinations. The model had replied exactly as instructed, "The provided documents do not answer this question." But retrieval confidence was high, so the system returned that as a normal answer.
*Fix:* that reply now becomes a structured abstention, a second layer behind the retrieval gate. This was the largest single gain: refusals went from 5 of 8 to 8 of 8.

**5. The abstain threshold was a guess (0.30).**
*Fix:* a calibration run with the threshold at 0 recorded, for every question, its retrieval confidence and what the model did. Every threshold was then replayed offline for free. In hybrid mode every threshold from 0.00 to 0.45 scored the same with zero wrongly refused questions. We chose the middle, **0.20**, for margin. At 0.20 the gate stops 4 of the 8 unanswerable questions before any LLM call, and the model's refusal handles the other 4.

**6. Code examples looked like uncited claims** (found by clicking around the dashboard after the final run). Every line of a code block counted as a sentence without a citation, so a correct, cited answer with a code example showed coverage 0.14; about a quarter of answers contain code.
*Fix:* code blocks are skipped when extracting claims, and they no longer split a paragraph, so "For example: <code> This makes q optional [1]." is one claim checked as a whole. A reminder that metrics need a human look, not just a number.

## What we learned about measuring

- **Temperature 0 is not deterministic.** Two runs of the identical configuration produced word-for-word identical answers for only 16 of 50 questions, and correctness differed by 0.02. With 50 questions, treat differences under about 0.03 as noise. That is why hybrid vs dense correctness (0.90 vs 0.89) is reported as a tie, and hybrid's advantage is claimed on ranking (MRR 0.80 vs 0.71) and on exact identifiers. For example, only BM25 found the passage for "What does `OAuth2PasswordBearer` do…", and dense mode abstained.
- **Keep the before numbers.** The baseline was committed before any fix, and nothing in `src/` changed while it ran, because a second process would have loaded half-fixed code.
- **Check the evaluation infrastructure too.** A network outage during the final run showed that judge failures, unlike pipeline failures, could crash a whole run. That is now recorded per question, and `--resume` re-runs only what failed.
- **Small tuning set.** The threshold was tuned on the same 50 questions it is reported on, and only 8 are unanswerable. One parameter limits the overfitting risk, but it should be re-checked on new questions.

## Chunking

Header-aware recursive chunks won on ranking (MRR 0.80 vs 0.58 for fixed windows) because each chunk is one topic and carries its heading breadcrumb. Semantic chunks trailed (correctness 0.87) because they lose that breadcrumb. See [CHUNKING.md](CHUNKING.md).

## Limits and next steps

- **Multi-hop and ambiguous questions** are the weak spots, and both are retrieval problems: the second relevant section is often not in the top 5 (recall@5 0.67 and 0.45). Next: split multi-part questions into sub-queries and rerank a wider candidate set.
- **Grow the golden set,** especially unanswerable questions, and re-check the 0.20 threshold on it.
- **Verify the Docker setup** on a machine with Docker, and add an OpenAI `seed` to reduce run-to-run variation.
