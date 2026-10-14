from langchain_openai import ChatOpenAI

from app import config


def get_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
        model=config.OPENAI_MODEL,
    )
