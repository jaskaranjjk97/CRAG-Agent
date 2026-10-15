from app.rag import build_bm25_index, hybrid_retrieve, load_chunks_from_disk

SAMPLES = [
    "What was Apple's total net sales in 2024?",
    "How many full-time employees does Microsoft have?",
    "What are Tesla's main risk factors?",
    "What was the revenue growth rate year over year?",
]


def main():
    # Pre-build BM25 for testing
    chunks = load_chunks_from_disk()
    build_bm25_index(chunks)

    for query in SAMPLES:
        print(f"\n{'=' * 60}\n> {query}")
        results = hybrid_retrieve(query=query, k=3)
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "?")
            snippet = doc.page_content[:150].replace("\n", " ")
            print(f"  [{i}] {source}")
            print(f"     {snippet}......")


if __name__ == "__main__":
    main()
