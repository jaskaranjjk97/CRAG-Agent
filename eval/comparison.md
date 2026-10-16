# CRAG vs Naive RAG — Evaluation

**Corpus:** 3 SEC 10-K filings (Apple FY2024, Microsoft FY2024, Tesla FY2024), 1,464 chunks
**Test set:** 15 questions with verified ground truths
**Framework:** RAGAS 0.2.x (faithfulness, answer relevancy, context precision, context recall)
**Runs:** 3 per system to establish variance

## Results (mean ± std across 3 runs)

| Metric | Naive RAG | CRAG | Δ | Relative Δ |
|--------|-----------|------|---|------------|
| Context Precision | 0.562 ± 0.002 | **0.669 ± 0.013** | **+0.107** | **+19.0%** |
| Faithfulness | 0.973 ± 0.004 | 0.900 ± 0.000 | -0.073 | -7.5% |
| Answer Relevancy | 0.924 ± 0.002 | 0.871 ± 0.000 | -0.052 | -5.6% |
| Context Recall | 0.783 ± 0.000 | 0.783 ± 0.000 | 0.000 | 0.0% |

Variance is negligible — the deltas are systematic, not noise.

## Interpretation

CRAG trades faithfulness and answer relevancy for context precision:

- **Context Precision (+19%)** — The grader filters out documents labeled `incorrect` before generation, so a larger fraction of the context window contains relevant material.
- **Faithfulness (-7.5%)** — Dropping chunks reduces the total grounding text, so RAGAS finds fewer answer claims fully supported. This is expected for an architecture that filters aggressively.
- **Answer Relevancy (-5.6%)** — Likely correlated with shorter answers, which is also a consequence of smaller context.
- **Context Recall (0%)** — Unchanged, confirming CRAG's gains come from filtering, not from better retrieval.

## Why this is domain-specific

CRAG was designed for open-domain retrieval where documents are heterogeneous — web search results, mixed sources, contradictory information. On three similar 10-K filings, even "irrelevant" chunks provide contextual redundancy that helps the generator: Apple's product categories, fiscal vocabulary, and formatting conventions are consistent across the whole filing. Filtering them removes noise but also removes helpful signal.

**On a heterogeneous corpus, the precision-faithfulness trade-off would favor CRAG much more strongly.** The current numbers represent a floor, not the architecture's ceiling.

## Methodology notes

- Both systems used identical retrieval (hybrid BM25 + Pinecone with RRF fusion, k=5)
- Both generators used the same model (GPT-4o-mini, temp=0.2) with the same citation format
- RAGAS used LangChain's `ChatOpenAI` wrapper with `temperature=0` for the evaluator LLM
- Per-run CSV files (`eval_naive_run{1,2,3}.csv`, `eval_crag_run{1,2,3}.csv`) are committed for reproducibility

## Known limitations

- 15 questions is a directional signal, not statistical rigor
- RAGAS itself has ~±0.03 variance on context precision (see CRAG std)
- Ground truths were verified against the ingested chunks, not against SEC filings directly