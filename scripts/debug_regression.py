"""Compare naive and CRAG answers for a question."""

import ast
import sys

import pandas as pd


def show_results(label, row):
    contexts = ast.literal_eval(row["retrieved_contexts"])

    print(f"\n### {label} ###")
    print(f"Faithfulness: {row['faithfulness']:.2f}")
    print(f"Context chunks: {len(contexts)}")

    print("\n--- Answer ---")
    print(row["response"])

    print("\n--- Contexts ---")
    for i, ctx in enumerate(contexts, start=1):
        ctx = str(ctx)

        print(f"\n[{i}] ({len(ctx)} chars)")
        print(ctx[:400])

        if len(ctx) > 400:
            print("...")


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    naive = pd.read_csv("eval/eval_naive.csv")
    crag = pd.read_csv("eval/eval_crag.csv")

    n = naive.iloc[idx]
    c = crag.iloc[idx]

    print("=" * 90)
    print(f"Question: {n['user_input']}")
    print("=" * 90)

    show_results("NAIVE", n)
    show_results("CRAG", c)


if __name__ == "__main__":
    main()
