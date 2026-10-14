from app import config


def main():
    print("Config loaded.")
    print(f"  OPENAI_MODEL: {config.OPENAI_MODEL}")
    print(f"  PINECONE_INDEX_NAME: {config.PINECONE_INDEX_NAME}")
    print(f"  TAVILY_API_KEY: {'set' if config.TAVILY_API_KEY else 'MISSING'}")
    print("Setup OK")


if __name__ == "__main__":
    main()
