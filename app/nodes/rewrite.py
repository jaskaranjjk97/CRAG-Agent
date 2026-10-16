from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import CRAGState

REWRITE_SYSTEM = """You rewrite the queries given by the user to get better 
retrieval quality over SEC - 10k filings.

The previous retrieval attempt returned only irrelevant documents. Your job
is to rewrite the query so the retriever finds better matches.

strategies:
- Add the entity name if the query is ambiguous ("Microsoft revenue" instead of "revenue")
- Add the fiscal year if not specified
- Use terms that appear in SEC filings ("net sales" instead of "revenue" if appropriate)
- Break a compound question into the primary one

Return ONLY the rewritten query. No explanation, no quotes.
"""


def rewrite_node(state: CRAGState) -> dict:
    """LangGraph node:rewrite the query for better retrieval."""

    original = (
        state.get("original_query")
        or state.get("query")
        or state["messages"][-1].content
    )
    current = state.get("query", original)
    retry_count = state.get("retry_count", 0)

    llm = get_llm(temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(
                content=(
                    f"Original query:{original}\n"
                    f"Previous attempt:{current}\n"
                    f"This is retry # {retry_count + 1}."
                )
            ),
        ]
    )

    new_query = response.content.strip().strip('"').strip("'")

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "rewrite",
            "status": "ok",
            "details": {
                "original_query": original,
                "previous_query": current,
                "new_query": new_query,
                "retry": retry_count + 1,
            },
        }
    )

    return {
        "query": new_query,
        "original_query": original,
        "retry_count": retry_count + 1,
        "trace": trace,
    }
