from langchain_core.messages import HumanMessage

from app.nodes.grader import grader_node
from app.rag import build_bm25_index, hybrid_retrieve, load_chunks_from_disk

chunks = load_chunks_from_disk()
build_bm25_index(chunks)

query = "What was Apple's total net sales in 2024?"
docs = hybrid_retrieve(query, k=4)

print(f"Retrieved: {len(docs)} docs")
for i, d in enumerate(docs):
    print(f"  [{i}] {d.metadata.get('source')} | {d.page_content[:80]!r}")

state = {
    "messages": [HumanMessage(content=query)],
    "retrieved_docs": docs,
    "trace": [],
}
result = grader_node(state)
print(f"\nGraded: {len(result['graded_docs'])} docs")
for g in result["graded_docs"]:
    print(f"  [{g['grade']}] {g['source']}")
