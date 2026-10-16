import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph import compiled_graph

app = FastAPI(
    title="CRAG Agent",
    description="Corrective RAG over SEC 10-K filings — hybrid retrieval, LLM-as-judge grading, web fallback, and query rewriting",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware(allow_origins="*", allow_methods="*", allow_headers="*")
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation ID; reuse across turns to keep context",
    )


class TraceEntry(BaseModel):
    node: str
    status: str
    details: dict = {}


class ChatResponse(BaseModel):
    thread_id: str
    query: str | None = None
    original_query: str | None = None
    final_response: str
    retry_count: int = 0
    web_search_used: bool = False
    grade_summary: dict | None = (
        None  # {correct, ambiguous, incorrect} from last grader run
    )
    trace: list[TraceEntry] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "service": "crag-agent",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run one turn through the CRAG graph."""
    try:
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=req.message)], "trace": []},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    except (RuntimeError, ValueError, KeyError, OSError, TimeoutError) as e:
        raise HTTPException(status_code=500, detail=f"graph_error: {e}")

    # Pull the last grade summary from the trace (most recent grader run)

    grade_summary = None
    trace = result.get("trace", [])
    reverse_trace = reversed(trace)

    for entry in reverse_trace:
        if entry.get("node") == "grader" and entry.get("status") == "ok":
            details = entry.get("details", {})
            grade_summary = details.get("summary")
            break

    return ChatResponse(
        thread_id=req.thread_id,
        query=result.get("query"),
        original_query=result.get("original_query"),
        final_response=result.get("final_response", ""),
        retry_count=result.get("retry_count", 0),
        web_search_used=result.get("web_search_used", False),
        grade_summary=grade_summary,
        trace=[
            TraceEntry(node=t["node"], status=t["status"], details=t.get("details", {}))
            for t in result.get("trace", [])
        ],
    )
