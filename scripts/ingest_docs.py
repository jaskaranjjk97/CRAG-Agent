import sys

from pinecone import Pinecone

from app import config
from app.rag import ingest_filings

if __name__ == "__main__":
    if "--force" in sys.argv:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        idx = pc.Index(config.PINECONE_INDEX_NAME)
        idx.delete(delete_all=True)
        print(f"Cleared all vectors from '{config.PINECONE_INDEX_NAME}'.")

    ingest_filings(force=True)
