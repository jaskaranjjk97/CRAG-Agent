from langchain_core.messages import HumanMessage

from app.nodes.grader import grader_node
from app.rag import build_bm25_index, hybrid_retrieve, load_chunks_from_disk

SAMPLES = [
    "What was Apple's total net sales in 2024?",
    "How many full-time employees does Microsoft have?",
    "What are Tesla's main risk factors?",
    "What was the revenue growth rate year over year?",
]


def main():
    # Warm up BM25
    chunks = load_chunks_from_disk()
    build_bm25_index(chunks)

    for query in SAMPLES:
        print(f"\n{'=' * 70}\n> {query}\n")

        docs = hybrid_retrieve(query, k=4)
        print(f"  → retrieved {len(docs)} docs")
        state = {
            "messages": [HumanMessage(content=query)],
            "retrieved_docs": docs,
            "trace": [],
        }
        result = grader_node(state)

        summary = result["trace"][-1]["details"]["summary"]
        print(f"  Summary: {summary}")
        for g in result["graded_docs"]:
            print(f"  [{g['grade']:>9}] {g['score']:.2f}  {g['source']}")
            print(f"              {g['reason']}")


if __name__ == "__main__":
    main()
