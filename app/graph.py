from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes.generate import generate_node
from app.nodes.grader import grader_node
from app.nodes.rewrite import rewrite_node
from app.nodes.router import route_after_grading
from app.nodes.web_fallback import web_fallback_node
from app.rag import hybrid_retrieve
from app.state import CRAGState


def retrieve_node(state: CRAGState) -> dict:
    """Wrapper node: run hybrid retrieval and store results."""
    query = state.get("query") or state["messages"][-1].content
    docs = hybrid_retrieve(query=query)

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "retrieve",
            "status": "ok",
            "details": {
                "query": query,
                "num_docs": len(docs),
                "source": list({d.metadata.get("source", "?") for d in docs}),
            },
        }
    )

    return {
        "query": query,
        "original_query": state.get("original_query") or query,
        "retrieved_docs": docs,
        "trace": trace,
    }


def build_graph():

    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grader", grader_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("rewrite_query", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grader")

    graph.add_conditional_edges(
        "grader",
        route_after_grading,
        {
            "generate": "generate",
            "web_fallback": "web_fallback",
            "rewrite_query": "rewrite_query",
        },
    )

    # web_fallback to generate node
    graph.add_edge("web_fallback", "generate")

    # rewrite node back to retrieve
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("generate", END)

    return graph.compile(checkpointer=MemorySaver())


compiled_graph = build_graph()
