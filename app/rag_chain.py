from __future__ import annotations
from functools import lru_cache

from app.config import settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

@lru_cache(maxsize=1)
def get_embeddings():
    if settings.EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbeddings(model=settings.EMBEDDINGS_PROVIDER, base_url=settings.OLLAMA_BASE_URL)
    
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OpenAI Embeddings key is not provided.")
    return OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL, api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_llm():
    if settings.LLM_PROVIDER.lower() == "ollama":
        return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_CHAT_MODEL)
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OpenAI API is not provided for LLM")
    return ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0)

def answer_question(question: str, context_pack: str) -> str:
    llm = get_llm()

    content = """
            "You are a helpful assistant answering questions using provided context.\n"
            "Rules:\n"
            "- Use ONLY the context. If missing, say you don't know.\n"
            "- Cite chunk_id/entity_id references when relevant.\n"
            "- Keep answers clear and concise.\n"
    """
    system = SystemMessage(content=content)

    human = HumanMessage(content=(
        f"QUESTION: \n{question}\n\n"
        f"CONTEXT: \n{context_pack}\n\n"
    ))

    resp = llm.invoke([system, human])
    return resp.content if hasattr(resp, "content") else str(resp)