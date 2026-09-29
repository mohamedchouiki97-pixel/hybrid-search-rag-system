# Chunking strategy comparison

50 questions, k = 5.

| Variant | Correctness | Faithfulness | Recall@k | MRR | Citation acc. | Abstain recall | Latency (s) |
|---|---|---|---|---|---|---|---|
| fixed | 0.83 | 0.95 | 0.74 | 0.58 | 0.82 | 0.62 | 3.21 |
| recursive | 0.85 | 0.93 | 0.81 | 0.79 | 0.88 | 0.62 | 2.75 |
| semantic | 0.77 | 0.90 | 0.72 | 0.66 | 0.92 | 0.50 | 2.72 |
