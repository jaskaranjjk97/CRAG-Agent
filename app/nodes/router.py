from app.state import CRAGState

MAX_RETRIES = 2


def route_after_grading(state: CRAGState) -> str:
    """Decide the next node based on graded document distribution.

    Returns one of: "generate", "web_fallback", "rewrite_query".
    """
    graded = state.get("graded_docs", [])
    retry_count = state.get("retry_count", 0)
    web_used = state.get("web_search_used", False)

    correct = sum(1 for g in graded if g["grade"] == "correct")
    ambiguous = sum(1 for g in graded if g["grade"] == "ambiguous")

    # 1. Any correct doc → generate directly.
    if correct >= 1:
        return "generate"

    # 2. Ambiguous docs → augment with web search (once).
    if ambiguous >= 1 and not web_used:
        return "web_fallback"

    # 3. Try rewriting the query (cheap, might unlock better local retrieval).
    if retry_count < MAX_RETRIES:
        return "rewrite_query"

    # 4. Rewrites exhausted, nothing useful found → try web as last resort.
    if not web_used:
        return "web_fallback"

    # 5. Everything tried → generate with whatever we have (graceful failure).
    return "generate"
