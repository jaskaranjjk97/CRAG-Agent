# Evaluation

This folder contains everything needed to reproduce the CRAG vs naive RAG benchmark.

## Files

| File | Purpose |
|------|---------|
| `questions.json` | 15-question eval set with verified ground truths |
| `eval_naive.csv` | Latest naive RAG run (per-question scores) |
| `eval_crag.csv` | Latest CRAG run |
| `eval_{system}_run{1,2,3}.csv` | 3-run variance study — the source of the stability numbers |
| `comparison.md` | Metric comparison, methodology, interpretation |

## Reproduce

From the repo root:

```bash
# Verify ground truths against ingested chunks
python -m scripts.verify_eval_set

# Run the eval (both systems, saves CSVs)
python -m scripts.evaluate

# Per-question diff between systems
python -m scripts.diagnose_eval

# 3-run variance analysis (requires run{1,2,3} CSVs)
python -m scripts.compare_runs