"""Verify ground-truth answers against the ingested filings.

For each question in data/eval_set.json, retrieve the top-3 chunks
and print them so you can eyeball whether the ground truth matches.
"""

import json
from pathlib import Path

from app.rag import build_bm25_index, hybrid_retrieve, load_chunks_from_disk


def main():
    eval_set = json.loads(Path("eval/questions.json").read_text())
    print(f"Loaded {len(eval_set)} questions. Building BM25 index...")
    chunks = load_chunks_from_disk()
    build_bm25_index(chunks)

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(eval_set)}] {q}")
        print(f"  GROUND TRUTH: {gt}")
        print(f"{'-' * 80}")

        docs = hybrid_retrieve(q, k=3)
        for j, d in enumerate(docs, 1):
            snippet = d.page_content[:300].replace("\n", " ")
            print(f"  [{j}] {d.metadata.get('source')}")
            print(f"      {snippet}")
        print()


if __name__ == "__main__":
    main()
