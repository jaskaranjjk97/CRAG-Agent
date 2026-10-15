from pinecone import Pinecone, ServerlessSpec

from app import config


def ensure_index_exits() -> None:
    """Checks whether the Pinecone index exist or not,create if not there."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = pc.list_indexes().names()

    if config.PINECONE_INDEX_NAME in existing:
        print(f"Index:{config.PINECONE_INDEX_NAME} already exists.Nothing to do.")
        return

    print(f"Creating index '{config.PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        ),
    )
    print(f"Index '{config.PINECONE_INDEX_NAME}' created.")


if __name__ == "__main__":
    ensure_index_exits()
