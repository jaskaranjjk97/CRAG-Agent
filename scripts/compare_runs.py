"""Compare faithfulness/relevancy/precision across multiple eval runs.

Usage:
    python -m scripts.compare_runs
"""

import pandas as pd


def load_run(system: str, run: int) -> pd.DataFrame:
    return pd.read_csv(f"eval/eval_{system}_run{run}.csv")


def main():
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    print(
        f"\n{'Metric':<22} {'System':<8} {'Run 1':>8} {'Run 2':>8} {'Run 3':>8} {'Mean':>8} {'Std':>8}"
    )
    print("-" * 78)

    for m in metrics:
        for system in ("naive", "crag"):
            vals = [load_run(system, r)[m].mean() for r in (1, 2, 3)]
            mean = sum(vals) / 3
            std = (sum((v - mean) ** 2 for v in vals) / 3) ** 0.5
            print(
                f"{m:<22} {system:<8} "
                f"{vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} "
                f"{mean:>8.3f} {std:>8.3f}"
            )
        print()

    # Delta analysis
    print("=" * 78)
    print("CRAG − Naive delta across runs:")
    print(
        f"{'Metric':<22} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10} {'Mean Δ':>10} {'Range':>10}"
    )
    print("-" * 78)
    for m in metrics:
        deltas = [
            load_run("crag", r)[m].mean() - load_run("naive", r)[m].mean()
            for r in (1, 2, 3)
        ]
        mean_d = sum(deltas) / 3
        rng = max(deltas) - min(deltas)
        print(
            f"{m:<22} {deltas[0]:>+10.3f} {deltas[1]:>+10.3f} {deltas[2]:>+10.3f} {mean_d:>+10.3f} {rng:>10.3f}"
        )


if __name__ == "__main__":
    main()
