from __future__ import annotations

#import os
#from dotenv import load_dotenv
#from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict


#load_dotenv()

#@dataclass
class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    NEO4J_URI: str #= os.getenv("NEO4J_URI")
    NEO4J_USER: str #= os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str #= os.getenv("NEO4J_PASSWORD")

    # Data
    DATA_DIR: str = "data" #= os.getenv("DATA_DIR")

    # retrievals
    DEFAULT_TOP_K: int = 5 #=os.getenv("DEFAULT_TOP_K")
    DEFAULT_HOPS: int = 1 #=os.getenv("DEFAULT_HOPS")

    # Embeddings
    #OLLAMA_EMBEDDINGS=os.getenv("OLLAMA_EMBEDDINGS")
    LLM_PROVIDER: str = "ollama" #=os.getenv("LLM_PROVIDER")
    EMBEDDINGS_PROVIDER:str = "ollama" #os.getenv("EMBEDDINGS_PROVIDER")

    # OpenAI
    OPENAI_API_KEY:str | None = None #=os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str | None = None
    OPENAI_EMBED_MODEL:str | None = None  #=os.getenv("OPENAI_EMBED_MODEL")
    OPENAI_CHAT_MODEL: str | None = None   #=os.getenv("OPENAI_CHAT_MODEL")
    AZURE_OPENAI_API_VERSION: str | None = None

    # Ollama
    OLLAMA_EMBEDDINGS: str | None = None
    OLLAMA_BASE_URL:str | None = None #=os.getenv("OLLAMA_BASE_URL")
    OLLAMA_EMBED_MODEL: str | None = None  #=os.getenv("OLLAMA_EMBED_MODEL")
    OLLAMA_CHAT_MODEL: str | None = None   #=os.getenv("OLLAMA_CHAT_MODEL")

    # Neo4j Embeddings
    VECTOR_INDEX_NAME: str = "chunk_vector_index"   #=os.getenv("VECTOR_INDEX_NAME")
    

settings = Settings()