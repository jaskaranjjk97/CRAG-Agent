## Evaluation

CRAG was benchmarked against a naive RAG baseline on 15 verified questions
over 3 SEC 10-K filings, using RAGAS across 3 runs per system.

| Metric | Naive RAG | CRAG | Δ |
|--------|-----------|------|---|
| **Context Precision** | 0.562 | **0.669** | **+0.107** |
| Faithfulness | 0.973 | 0.900 | -0.073 |
| Answer Relevancy | 0.924 | 0.871 | -0.052 |
| Context Recall | 0.783 | 0.783 | 0.000 |

CRAG improves context precision by 19% by filtering out documents graded as
irrelevant. The cost is a small loss in faithfulness (-7.5%) because filtering
reduces the total grounding text. This is the expected trade-off on a
**homogeneous corpus** — three similar filings where even "irrelevant" chunks
provide contextual redundancy. On heterogeneous sources (web results, mixed
documents), the precision gain would be larger and the faithfulness cost
smaller.

[Full methodology, per-question results, and raw CSVs →](eval/comparison.md)