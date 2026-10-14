from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Grade = Literal["correct", "ambiguous", "incorrect"]


class CRAGState(TypedDict, total=False):
    # Conversation
    messages: Annotated[BaseMessage, add_messages]

    # Retrieval part
    query: str
    retrieved_docs: list[Document]
    graded_docs: list[dict]  # [{"doc":Document,"grade":"incorrect","score":0.80}]
    web_results: list[Document]

    # Generation
    final_response: str

    # Observability
    trace: list[dict]
