from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import CRAGState, Grade


class DocGrade(BaseModel):
    grade: Grade = Field(description="One of: correct, ambiguous, incorrect")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    reason: str = Field(description="reason for the grade assigned")


GRADER_SYSTEM = """You are a relevance grader for a RAG system over SEC 10-K filings.

Given a user query and a single retrieved document, decide how relevant
that document is for answering the query.

Grades:
- "correct": The document contains information that directly answers the
  query. You would use this document to write the answer.
- "ambiguous": The document is on-topic but only partially answers the
  query, or is missing the specific detail asked for.
- "incorrect": The document is off-topic, OR it is about a different
  company/entity/year than the query asks about, OR it is temporally
  incompatible with the query.

Rules:
- Grade against the SPECIFIC query, not general topical similarity.
- Entity match matters. If the query is about Tesla and the document is
  from Apple's 10-K, grade it "incorrect" — even if the topic matches.
- Temporal match matters. A 10-K filing is a historical document frozen
  at its filing date. If the query asks about "today", "current", "now",
  "recent", "latest", or a future date, the filing CANNOT answer it —
  grade as "incorrect" regardless of how relevant the topic appears.
  Historical stock prices in a filing do NOT answer a query about today's
  stock price.
- Filings are long and repetitive across companies. "Risk factors" appears
  in every 10-K, so topical similarity alone is not enough.
- Return a relevance score (0-1) with the grade and a one-sentence reason.
- If the query asks about a specific historical date or year that the
  filing covers, grade the document normally — the temporal rule only
  applies to present-tense queries ("today", "current", "now").
"""


def _grade_one(query: str, doc_text: str, source: str, ticker: str) -> DocGrade:
    """Grade a single document against the query."""
    llm = get_llm(temperature=0).with_structured_output(DocGrade)

    prompt = f"""query:{query}
Document source: {source}
Entity ticker: {ticker}

Document excerpt:
\"\"\"
{doc_text[:1500]}
\"\"\"

Grade this document for the query."""

    return llm.invoke(
        [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=prompt)]
    )


def grader_node(state: CRAGState) -> dict:
    """Langchain Node:grade each retrieved document for the query."""
    query = state.get("query") or state["messages"][-1].content
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {
            "graded_docs": [],
            "trace": state.get("trace", [])
            + [
                {
                    "node": "grader",
                    "status": "no_docs",
                    "details": {"message": "No Documents to Grade."},
                }
            ],
        }

    graded: list[dict] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        ticker = doc.metadata.get("ticker", "unknown")
        result = _grade_one(query, doc.page_content, source, ticker)

        graded.append(
            {
                "doc": doc,
                "grade": result.grade,
                "score": result.score,
                "reason": result.reason,
                "source": source,
            }
        )

    # Summary counts for the router
    summary = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    for g in graded:
        summary[g["grade"]] += 1

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "grader",
            "status": "ok",
            "details": {
                "summary": summary,
                "grades": [
                    {
                        "source": g["source"],
                        "grade": g["grade"],
                        "score": round(g["score"], 2),
                        "reason": g["reason"],
                    }
                    for g in graded
                ],
            },
        }
    )

    return {
        "query": query,
        "graded_docs": graded,
        "trace": trace,
    }
