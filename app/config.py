from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Settings:
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")

    # Data
    DATA_DIR = os.getenv("DATA_DIR")

    # retrievals
    DEFAULT_TOP_K=os.getenv("DEFAULT_TOP_K")
    DEFAULT_HOPS=os.getenv("DEFAULT_HOPS")

    # Embeddings
    OLLAMA_EMBEDDINGS=os.getenv("OLLAMA_EMBEDDINGS")
    LLM_PROVIDER=os.getenv("LLM_PROVIDER")

    # OpenAI
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    OPENAI_EMBED_MODEL=os.getenv("OPENAI_EMBED_MODEL")
    OPENAI_CHAT_MODEL=os.getenv("OPENAI_CHAT_MODEL")

    # Ollama
    OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL")
    OLLAMA_EMBED_MODEL=os.getenv("OLLAMA_EMBED_MODEL")
    OLLAMA_CHAT_MODEL=os.getenv("OLLAMA_CHAT_MODEL")

    # Neo4j Embeddings
    VECTOR_INDEX_NAME=os.getenv("VECTOR_INDEX_NAME")

settings = Settings()