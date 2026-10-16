from app.state import CRAGState

MAX_RETRIES = 2


def route_after_grading(state: CRAGState) -> str:
    """Decide the next node based on graded document distribution.

    Returns one of : "web_fallback" , "generate" , "rewrite_query" .
    """

    graded = state.get("graded_docs", [])
    retry_count = state.get("retry_count", 0)
    web_used = state.get("web_search_used", False)

    correct = sum(1 for g in graded if g["grade"] == "correct")
    ambiguous = sum(1 for g in graded if g["grade"] == "ambiguous")

    # If we have at least one clearly-correct document, generate directly.
    if correct >= 1:
        return "generate"

    # If nothing is correct but something is ambiguous, try web fallback
    # (but only if we haven't already done it).
    if ambiguous >= 1 and not web_used:
        return "web_fallback"

    # Nothing useful at all — try rewriting the query, up to MAX_RETRIES.
    if retry_count < MAX_RETRIES:
        return "rewrite_query"

    # Exhausted all options — generate from whatever we have (or say we can't).
    return "generate"
