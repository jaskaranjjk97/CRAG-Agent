"""Per-question diff between naive RAG and CRAG.

Reads the RAGAS output CSVs and prints:
- Per-question deltas for faithfulness, context precision, context recall
- Distribution statistics
- The three biggest CRAG regressions with their questions
"""

import pandas as pd


def main():
    naive = pd.read_csv("eval/eval_naive.csv")
    crag = pd.read_csv("eval/eval_crag.csv")

    q_col = "user_input"

    print(f"\n{'#':>3} {'Question':<50} {'Faith Δ':>9} {'Prec Δ':>9} {'Recall Δ':>10}")
    print("-" * 88)

    for i in range(len(naive)):
        q = str(naive.iloc[i][q_col])[:46]
        faith_d = crag.iloc[i]["faithfulness"] - naive.iloc[i]["faithfulness"]
        prec_d = crag.iloc[i]["context_precision"] - naive.iloc[i]["context_precision"]
        rec_d = crag.iloc[i]["context_recall"] - naive.iloc[i]["context_recall"]
        marker = "  <-- REGRESSION" if faith_d < -0.2 else ""
        print(
            f"{i + 1:>3} {q:<50} {faith_d:>+9.2f} {prec_d:>+9.2f} {rec_d:>+10.2f}{marker}"
        )

    print("\n" + "=" * 88)
    print("Distribution of faithfulness deltas (CRAG - naive):")
    deltas = crag["faithfulness"] - naive["faithfulness"]
    print(f"  Mean:    {deltas.mean():+.3f}")
    print(f"  Median:  {deltas.median():+.3f}")
    print(f"  Std:     {deltas.std():.3f}")
    print(f"  Worse (Δ < -0.2):  {(deltas < -0.2).sum()} questions")
    print(f"  Better (Δ > 0.2):  {(deltas > 0.2).sum()} questions")
    print(
        f"  Similar:           {((deltas >= -0.2) & (deltas <= 0.2)).sum()} questions"
    )

    print("\n" + "=" * 88)
    print("Biggest CRAG regressions (faithfulness):")
    combined = (
        pd.DataFrame(
            {
                "question": naive[q_col],
                "naive_faith": naive["faithfulness"],
                "crag_faith": crag["faithfulness"],
                "delta": crag["faithfulness"] - naive["faithfulness"],
            }
        )
        .sort_values("delta")
        .head(5)
    )

    for _, row in combined.iterrows():
        print(f"\n  Q: {row['question'][:72]}")
        print(
            f"     naive: {row['naive_faith']:.2f}  →  crag: {row['crag_faith']:.2f}   (Δ {row['delta']:+.2f})"
        )

    # Also show the actual retrieved context sizes for the worst offenders
    print("\n" + "=" * 88)
    print(
        "Context size comparison (how many chunks each system passed to the generator):"
    )
    for i, row in combined.iterrows():
        idx = naive[naive[q_col] == row["question"]].index[0]
        n_ctx = len(eval(naive.iloc[idx]["retrieved_contexts"]))
        c_ctx = len(eval(crag.iloc[idx]["retrieved_contexts"]))
        print(f"  {n_ctx} vs {c_ctx} chunks — {row['question'][:60]}")


if __name__ == "__main__":
    main()
