import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from app import config

# ---------------------------------------------------------------------------
# In-memory BM25 index
# ---------------------------------------------------------------------------

_bm25_index: BM25Okapi | None = None
_bm25_docs: list[Document]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Adequate for BM25."""
    return text.lower().split()


def build_bm25_index(docs: list[Document]) -> None:
    global _bm25_index, _bm25_docs
    tokenized = [_tokenize(d.page_content) for d in docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_docs = docs
    print(f"BM25 index build with {len(docs)} documents.")


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        dimensions=config.EMBEDDING_DIMENSION,
        api_key=config.OPENAI_API_KEY,
    )


def _get_vectorstore() -> PineconeVectorStore:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    return PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_filings(force: bool = False) -> list[Document]:
    """Chunk filings, embed them, and upsert to Pinecone.

    Also returns the chunks so the caller can build a BM25 index.

    """
    from app.parsers import load_all_filings

    raw_docs = load_all_filings()

    if not raw_docs:
        print("No filing data found in data/filings.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split {len(raw_docs)} filings into {len(chunks)} chunks.")

    vectorstore = _get_vectorstore()

    if not force:
        existing = vectorstore.similarity_search("text", k=1)
        if existing:
            print("Pinecone index already populated no need to upsert again")
            return chunks

    vectorstore.add_documents(chunks)
    print(
        f"Upserted {len(chunks)} chunks to Pinecone index '{config.PINECONE_INDEX_NAME}'."
    )

    # Build B25 index for the same chunks
    build_bm25_index(chunks)

    return chunks


def load_chunks_from_disk() -> list[Document]:
    """For Lazy initialization of BM25 we are loading the docs again
    and then rebuilding index as in memory bm25 index get lost."""

    from app.parsers import load_all_filings

    raw_docs = load_all_filings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(raw_docs)


# ---------------------------------------------------------------------------
# Hybrid retrieval with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _dense_retrieve(query: str, k: int) -> list[Document]:
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(
        query=query,
        k=k,
    )


def _sparse_retrieve(query: str, k: int) -> list[Document]:
    global _bm25_docs

    if _bm25_index is None:
        _bm25_docs = load_chunks_from_disk()
        build_bm25_index(_bm25_docs)

    tokenize_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenize_query)

    # Get top-k indices by score

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [_bm25_docs[i] for i in top_indices]


def _reciprocal_rank_fusion(
    dense: list[Document],
    sparce: list[Document],
    k: int = 60,
) -> list[Document]:
    """Fuse two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1 / (k + rank)) across lists.
    k=60 is the standard constant from the original paper.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _key(doc: Document) -> str:
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    for ranked_list in (dense, sparce):
        for rank, doc in enumerate(ranked_list, start=1):
            key = _key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in fused]


def hybrid_retrieve(query: str, k: int = 5) -> list[Document]:
    """Hybrid retrieval: BM25 + dense, fused with RRF.

    Retrieves k*2 from each source, then fuses and returns top-k.
    """
    dense_docs = _dense_retrieve(query, k=k * 2)
    sparse_docs = _sparse_retrieve(query, k=k * 2)
    fused = _reciprocal_rank_fusion(dense_docs, sparse_docs, k=60)
    return fused[:k]
