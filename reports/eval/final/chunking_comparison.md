# Chunking strategy comparison

50 questions, k = 5.

| Variant | Correctness | Faithfulness | Recall@k | MRR | Citation acc. | Abstain recall | Latency (s) |
|---|---|---|---|---|---|---|---|
| fixed | 0.92 | 0.99 | 0.74 | 0.58 | 0.96 | 1.00 | 4.15 |
| recursive | 0.92 | 0.98 | 0.81 | 0.80 | 0.96 | 1.00 | 4.16 |
| semantic | 0.87 | 1.00 | 0.72 | 0.66 | 0.97 | 1.00 | 3.67 |
