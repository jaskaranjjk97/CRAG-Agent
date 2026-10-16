import uuid

from langchain_core.messages import HumanMessage

from app.graph import compiled_graph

SAMPLES = [
    "What was Apple's total net sales in 2024?",
    "How many full-time employees does Microsoft have?",
    "What are Tesla's main risk factors?",
    "What is Apple's stock price today?",  # ← forces web fallback (recency)
    "What was revenue last year?",  # ← forces rewrite (no entity)
    "How many vacation days do Microsoft employees get?",
]


def run(query: str) -> None:
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "trace": [],
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    print(f"\n{'=' * 70}")
    print(f"> {query}")
    print(f"  final_query: {result.get('query')}")
    print(f"  retries: {result.get('retry_count', 0)}")
    print(f"  web_used: {result.get('web_search_used', False)}")
    print(f"  trace: {[t['node'] for t in result.get('trace', [])]}")
    print(f"  answer: {result.get('final_response', '')[:300]}...")


def main():
    for q in SAMPLES:
        run(q)


if __name__ == "__main__":
    main()
