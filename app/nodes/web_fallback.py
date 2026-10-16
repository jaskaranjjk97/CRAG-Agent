from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from app import config
from app.state import CRAGState


def web_fallback_node(state: CRAGState) -> dict:
    """LangGraph node:if LLM returns ambiguous this node will lead to web_search."""

    query = state.get("query") or state["messages"][-1].content

    try:
        search = TavilySearch(
            api_key=config.TAVILY_API_KEY,
            max_results=4,
        )
        raw_results = search.invoke({"query": query})
    except (OSError, ValueError, RuntimeError) as e:
        return {
            "web_results": [],
            "web_search_used": True,
            "trace": state.get("trace", [])
            + [
                {
                    "node": "web_fallback",
                    "status": "error",
                    "details": {"error": str(e), "error_type": type(e).__name__},
                }
            ],
        }

    # Tavily returns a list of dicts (or a wrapper dict depending on version)
    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    else:
        results = raw_results

    docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("url", "unknown"),
                "title": r.get("title", ""),
                "origin": "tavily",
            },
        )
        for r in results
    ]
    trace = state.get("trace", [])
    trace.append(
        {
            "node": "web_fallback",
            "status": "ok",
            "details": {
                "num_results": len(docs),
                "urls": [d.metadata["source"] for d in docs],
            },
        }
    )

    return {
        "web_results": docs,
        "web_search_used": True,
        "trace": trace,
    }
