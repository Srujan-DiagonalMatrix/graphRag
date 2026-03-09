from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any

class HealthResponse(BaseModel):
    ok: bool
    neo4j: bool
    message: str
    vector_index: bool | None = None

class IngestRequest(BaseModel):
    files: list[str] | None = None
    chunk_size: int = Field(default=800, ge=200, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

class IngestResponse(BaseModel):
    ok: bool
    entitles_created: int
    relationships_created: int
    chunks_created: int
    embedding_dim: int
    vector_index: str
    message: str

class Queryrequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    hops: int = Field(default=1, ge=0, le=3)

class SourceItem(BaseModel):
    chunk_id: str
    entity_id: str | None = None
    score: float | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    graph_paths: list[dict[str, Any]] = []
    used_top_k: int
    used_hops: int