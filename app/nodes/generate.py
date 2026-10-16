from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import CRAGState

GENERATE_SYSTEM = """You are a helpful assistant answering questions about SEC 10-K filings.

Use ONLY the provided context to answer the query. If the context does not
contain enough information, say so clearly — do not make up facts.

Rules:
- Cite the source of each claim inline (e.g., "According to apple-10k-2024.htm, ...").
- If sources conflict, note the conflict rather than picking one silently.
- If the context is web results, cite the URL.
- Keep answers short and direct. No filler.
"""


def generate_node(state: CRAGState) -> dict:
    """LangGraph node: uses LLM to generate the final_response."""

    query = state.get("query") or state["messages"][-1].content
    graded_results = state.get("graded_docs", [])
    web_results = state.get("web_results", [])

    # Keep only the usable chunks("correct","ambiguous")
    usable = [
        g["doc"] for g in graded_results if g["grade"] in ("correct", "ambiguous")
    ]
    usable.extend(web_results)

    if not usable:
        return {
            "final_response": (
                "I couldn't find reliable information to answer that question."
                "Try rephrasing, or ask about a specific company and fiscal year."
            ),
            "trace": state.get("trace", [])
            + [
                {
                    "node": "generate",
                    "status": "no_context",
                    "details": {"message": "No usable context found for the query."},
                }
            ],
        }

    # Build context blocks with source label
    blocks = []
    for i, doc in enumerate(usable, 1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] Source:{source}\n {doc.page_content[:1200]}")
    context = "\n\n-----\n\n".join(blocks)

    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=f"Context:\n\n {context}\n\n Query:{query} "),
        ]
    )

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "generate",
            "status": "ok",
            "details": {
                "num_docs_used": len(usable),
                "used_web_results": len(web_results),
                "used_local_docs": len(usable) - len(web_results),
            },
        }
    )

    return {
        "final_response": response.content,
        "trace": trace,
    }
