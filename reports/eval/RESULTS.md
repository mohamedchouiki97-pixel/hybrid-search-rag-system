# Evaluation: baseline → final

50 golden questions, gpt-4.1-mini-2025-04-14 for generation and judging. fixed / recursive / semantic use hybrid retrieval; hybrid / dense use recursive chunks.

| Variant | Correctness | Faithfulness | Recall@5 | MRR | Citation acc. | Abstain recall | Abstain precision |
|---|---|---|---|---|---|---|---|
| fixed | 0.83 → **0.92** (+0.09) | 0.95 → **0.99** (+0.05) | 0.74 (=) | 0.58 (=) | 0.82 → **0.96** (+0.13) | 0.62 → **1.00** (+0.38) | 0.83 → **1.00** (+0.17) |
| recursive | 0.85 → **0.92** (+0.07) | 0.93 → **0.98** (+0.06) | 0.81 (=) | 0.80 (=) | 0.88 → **0.96** (+0.08) | 0.62 → **1.00** (+0.38) | 1.00 (=) |
| semantic | 0.77 → **0.87** (+0.10) | 0.90 → **1.00** (+0.10) | 0.72 (=) | 0.66 (=) | 0.92 → **0.97** (+0.06) | 0.50 → **1.00** (+0.50) | 0.80 → **1.00** (+0.20) |
| hybrid | 0.85 → **0.90** (+0.05) | 0.91 → **0.98** (+0.08) | 0.81 (=) | 0.80 (=) | 0.87 → **0.97** (+0.10) | 0.62 → **1.00** (+0.38) | 1.00 (=) |
| dense | 0.74 → **0.89** (+0.15) | 0.79 → **1.00** (+0.21) | 0.77 (=) | 0.71 (=) | 0.88 → **0.95** (+0.07) | 0.00 → **1.00** (+1.00) | n/a |

## By question type (recursive, hybrid)

| Type | n | Correctness | Faithfulness | Recall@5 | Citation acc. |
|---|---|---|---|---|---|
| lookup | 25 | 0.98 (=) | 0.99 → **0.98** (-0.01) | 0.96 (=) | 0.85 → **0.94** (+0.09) |
| multi_hop | 10 | 0.80 → **0.85** (+0.05) | 1.00 (=) | 0.67 (=) | 0.88 → **1.00** (+0.12) |
| no_answer | 8 | 0.62 → **1.00** (+0.38) | n/a | n/a | n/a |
| ambiguous | 7 | 0.71 (=) | 1.00 → **0.96** (-0.04) | 0.45 (=) | 0.97 → **0.96** (-0.01) |

## Reading these numbers

- **What changed between the runs:** claim grouping for citation coverage, one judge call per claim with all its cited passages, cross-encoder confidence in both retrieval modes (best chunk, not first), the model's "documents do not answer" reply turned into a structured abstention, and the abstain threshold set to 0.20 from a calibration sweep (`threshold_sweep.md`).
- **Retrieval metrics are unchanged on purpose.** No fix touched retrieval, so identical Recall@5 and MRR are a sanity check that the comparison is fair.
- **Run-to-run noise is about ±0.02–0.04 on correctness.** Even at temperature 0 only 16 of 50 answers were word-for-word identical between two runs of the same configuration (recursive and hybrid are the same setup and scored 0.92 and 0.90). Differences smaller than that, such as hybrid vs dense correctness (0.90 vs 0.89), are not meaningful; hybrid's real advantage shows in retrieval (MRR 0.80 vs 0.71).
- **Abstention now works in both layers.** The model's own refusal catches every unanswerable question; the 0.20 retrieval gate stops 4 of the 8 before any LLM call (q042 scores 0.24 and is caught by the model instead). Abstention precision is 1.00: no answerable question was refused.
- **Small tuning set.** The threshold was chosen on the same 50 questions reported here, and only 8 are unanswerable. Re-check it when the golden set grows.
- **Weak spots left:** ambiguous questions (retrieval recall 0.45) and multi-hop questions (0.67). Both are retrieval problems: the right second section is not in the top 5.
- **Found after the final run, in manual testing:** answers containing a code example had understated citation coverage, because every code line counted as an uncited claim (a correct answer scored 0.14). That affected 10-14 of about 40 answered questions per configuration. Fixed in `6e82969`. Coverage feeds the confidence shown to users, not any metric in the tables above.
- **Latency** is not comparable across these runs: the final run overlapped a network outage and its recovery.
