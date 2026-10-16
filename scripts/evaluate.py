import json
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app import config
from app.graph import compiled_graph
from app.llm import get_llm
from app.rag import build_bm25_index, hybrid_retrieve, load_chunks_from_disk

# ---------------------------------------------------------------------------
# Naive RAG baseline
# ---------------------------------------------------------------------------


NAIVE_SYSTEM = """You are a helpful assistant answering questions about SEC 10-K filings.

Use ONLY the provided context to answer the query. If the context does not
contain enough information, say so clearly. Cite sources inline."""


def naive_rag(query: str) -> dict:
    """Top-K retrieve → generate. No grading, no correction."""

    docs = hybrid_retrieve(query=query, k=5)

    context_block = [
        f"[{i}] {d.metadata.get('source', '?')}\n{d.page_content[:1200]}"
        for i, d in enumerate(docs, 1)
    ]
    context = "\n\n----\n\n".join(context_block)
    llm = get_llm(temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(content=NAIVE_SYSTEM),
            HumanMessage(content=(f"Contetx:{context}\n Query:{query}.")),
        ]
    )

    return {"answer": response.content, "context": [d.page_content for d in docs]}


# ---------------------------------------------------------------------------
# CRAG
# ---------------------------------------------------------------------------


def crag(query: str) -> dict:
    """Run the full CRAG graph."""
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "trace": [],
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    # Extract only the docs the generator actually used
    graded = result.get("graded_docs", [])
    used = [g["doc"] for g in graded if g["grade"] in ("correct", "ambiguous")]
    used.extend(result.get("web_results", []))

    return {
        "answer": result.get("final_response", []),
        "context": [doc.page_content for doc in used] if used else [""],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval(system_name: str, fn, eval_set: list[dict]) -> dict:
    print(f"\nRunning {system_name}.....")
    answers, contexts, gts = [], [], []

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        print(f"  [{i}/{len(eval_set)}] {q}")
        result = fn(q)
        answers.append(result["answer"])
        contexts.append(result["context"])
        gts.append(item["ground_truth"])

    ds = Dataset.from_dict(
        {
            "question": [x["question"] for x in eval_set],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": gts,
        }
    )

    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL,
        )
    )

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    return result.to_pandas()


def main():
    eval_path = Path("eval/questions.json")
    eval_set = json.loads(eval_path.read_text())

    print(f"Loaded {len(eval_set)} evaluation questions.")

    # warm up the BM25 index
    chunks = load_chunks_from_disk()
    build_bm25_index(chunks)

    naive_df = run_eval("Naive_RAG", naive_rag, eval_set)
    crag_df = run_eval("CRAG", crag, eval_set)

    # Save raw results
    naive_df.to_csv("eval/eval_naive.csv", index=False)
    crag_df.to_csv("eval/eval_crag.csv", index=False)

    # Print comparison
    print("\n" + "=" * 70)
    print("METRIC COMPARISON (mean across questions)")
    print("=" * 70)

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    print(f"{'Metric':<25} {'Naive RAG':>12} {'CRAG':>12} {'Δ':>10}")
    print("-" * 70)

    for m in metrics:
        n = naive_df[m].mean()
        c = crag_df[m].mean()
        delta = c - n
        sign = "+" if delta >= 0 else ""

        print(f"{m:<25} {n:>12.3f} {c:>12.3f} {sign}{delta:>9.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
