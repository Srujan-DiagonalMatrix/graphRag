from __future__ import annotations
from functools import lru_cache

from app.config import settings

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI

@lru_cache(maxsize=1)
def get_embeddings():
    if settings.EMBEDDINGS_PROVIDER.lower() == "ollama":
        return OllamaEmbeddings(model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)
    
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OpenAI Embeddings key is not provided.")
    return OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL, api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_llm():

    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_CHAT_MODEL)
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not provided by LLM.")
        
        base_url = (settings.OPENAI_BASE_URL or "").rstrip("/")

        if "openai.azure.com" in base_url:
            if not settings.AZURE_OPENAI_API_VERSION:
                raise RuntimeError("AZURE_OPENAI_API_VERSION is a required value and missing.")
            if not settings.OPENAI_CHAT_MODEL:
                raise RuntimeError("OPENAI_CHAT_MODEL must be your azure deployment model.")
            
            return AzureChatOpenAI(azure_endpoint=base_url, 
                                   api_version=settings.AZURE_OPENAI_API_VERSION, 
                                   azure_deployment=settings.OPENAI_CHAT_MODEL, 
                                   temperature=0, 
                                   api_key=settings.OPENAI_API_KEY)
        
        return ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, 
                          temperature=0, 
                          base_url=base_url, 
                          api_key=settings.OPENAI_API_KEY)
    
    raise RuntimeError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


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