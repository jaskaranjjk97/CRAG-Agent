import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CRAG Agent",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_agent(message: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "thread_id": st.session_state.thread_id},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _node_icon_label(node: str) -> tuple[str, str]:
    return {
        "retrieve": ("🔍", "retrieve"),
        "grader": ("⚖️", "grader"),
        "web_fallback": ("🌐", "web_fallback"),
        "rewrite": ("✏️", "rewrite_query"),
        "generate": ("✨", "generate"),
    }.get(node, ("⚙️", node))


def _status_icon(status: str) -> str:
    return {
        "ok": "✅",
        "error": "❌",
        "no_docs": "⚠️",
        "no_context": "⚠️",
        "unexpected_shape": "⚠️",
    }.get(status, "•")


def _render_details(node: str, status: str, details: dict) -> None:
    """Render node-specific detail blocks."""

    if node == "retrieve":
        st.markdown(f"**Query:** `{details.get('query', '')}`")
        st.markdown(f"**Documents retrieved:** {details.get('num_docs', 0)}")
        sources = details.get("sources", [])
        if sources:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in sources))

    elif node == "grader":
        summary = details.get("summary", {})
        cols = st.columns(3)
        cols[0].metric("✅ Correct", summary.get("correct", 0))
        cols[1].metric("⚠️ Ambiguous", summary.get("ambiguous", 0))
        cols[2].metric("❌ Incorrect", summary.get("incorrect", 0))

        grades = details.get("grades", [])
        if grades:
            st.markdown("**Per-document grades:**")
            for g in grades:
                grade = g.get("grade", "?")
                badge = {"correct": "✅", "ambiguous": "⚠️", "incorrect": "❌"}.get(
                    grade, "•"
                )
                st.markdown(
                    f"{badge} **{grade}** · `{g.get('score', 0):.2f}` · `{g.get('source', '?')}`  \n"
                    f"   _{g.get('reason', '')}_"
                )

    elif node == "web_fallback":
        if status == "error":
            st.error(f"Web fallback failed: {details.get('error', 'unknown')}")
        else:
            st.markdown(f"**Results:** {details.get('num_results', 0)}")
            urls = details.get("urls", [])
            for u in urls:
                st.markdown(f"- [{u}]({u})")

    elif node == "rewrite":
        st.markdown("**Original:** " + details.get("original_query", ""))
        st.markdown("**Previous attempt:** " + details.get("previous_query", ""))
        st.markdown("**New query:** " + details.get("new_query", ""))
        st.caption(f"Retry #{details.get('retry', '?')}")

    elif node == "generate":
        if status == "no_context":
            st.warning(details.get("message", "No usable context."))
        else:
            st.markdown(f"**Docs used:** {details.get('num_docs_used', 0)}")
            st.caption(
                f"Local: {details.get('used_local_docs', 0)} · "
                f"Web: {details.get('used_web_results', 0)}"
            )

    else:
        st.json(details)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_session = st.columns([3, 1])
with col_title:
    st.title("🔎 CRAG Agent")
    st.caption(
        "Corrective RAG over SEC 10-K filings — hybrid retrieval · LLM-as-judge · web fallback · query rewrite"
    )
with col_session:
    st.caption(f"**Session:** `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.last_trace = []
        st.session_state.last_meta = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

chat_col, trace_col = st.columns([3, 2])


# ----------------------- Left: Chat -----------------------

with chat_col:
    st.subheader("💬 Ask")

    chat_container = st.container(height=520, border=False)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask about the filings, e.g. 'What was Apple's iPhone revenue in 2024?'"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("CRAG is retrieving, grading, and correcting..."):
            result = call_agent(user_input)

        if result:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result.get("final_response", "(no response)"),
                }
            )
            st.session_state.last_trace = result.get("trace", [])
            st.session_state.last_meta = {
                "query": result.get("query"),
                "original_query": result.get("original_query"),
                "retry_count": result.get("retry_count", 0),
                "web_search_used": result.get("web_search_used", False),
                "grade_summary": result.get("grade_summary"),
            }

        st.rerun()


# ----------------------- Right: Trace panel -----------------------

with trace_col:
    st.subheader("🔍 Pipeline Trace")

    meta = st.session_state.last_meta

    if meta:
        # Summary metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Rewrites", meta.get("retry_count", 0))
        with m2:
            st.metric(
                "Web search", "🌐 Yes" if meta.get("web_search_used") else "✅ No"
            )
        with m3:
            gs = meta.get("grade_summary")
            if gs:
                st.metric(
                    "Docs graded",
                    f"{gs.get('correct', 0)}/{gs.get('ambiguous', 0)}/{gs.get('incorrect', 0)}",
                )
            else:
                st.metric("Docs graded", "—")

        # Query change indicator (if rewrite happened)
        oq = meta.get("original_query")
        cq = meta.get("query")
        if oq and cq and oq != cq:
            st.info(f"**Query rewritten**\n\n**Original:** {oq}\n\n**Final:** {cq}")

        st.divider()

    if not st.session_state.last_trace:
        st.info("Ask a question to see the CRAG pipeline trace here.")
    else:
        for i, step in enumerate(st.session_state.last_trace, 1):
            node = step.get("node", "unknown")
            status = step.get("status", "ok")
            details = step.get("details", {})

            icon, label = _node_icon_label(node)
            status_icon = _status_icon(status)

            with st.expander(
                f"{icon} **Step {i}: {label}** — {status_icon} `{status}`",
                expanded=(i == len(st.session_state.last_trace)),
            ):
                _render_details(node, status, details)
