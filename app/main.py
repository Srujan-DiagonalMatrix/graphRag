from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.neo4j_client import Neo4jClient

from app.schemas import (HealthResponse, IngestRequest, IngestResponse, Queryrequest, SourceItem, QueryResponse)
from app.config import settings
from app.ingest import ingest_to_neo4j
from app.retriever import vector_retrieve, graph_expand, build_context_pack
from app.rag_chain import get_embeddings, answer_question

app = FastAPI(title="GraphRAG MVP", version="0.1.0")

app.middleware(
    CORSMiddleware,
    allow_origin=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_db() -> Neo4jClient:
    return Neo4jClient()

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db = get_db()

    try:
        ok = db.ping()
        idx = db.vector_index_exists(settings.VECTOR_INDEX_NAME)        
        return HealthResponse(ok=ok,neo4j=ok,vector_index=idx, message="Neo4j is reachable" if ok else "Neo4j is not reachable.")
    finally:
        db.close()

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    db = get_db()
    try:
        if not db.ping():
            raise HTTPException(status_code=503, detail="Neo4j is not accessible.")
        
        result = ingest_to_neo4j(neo4j=db,
                        files=req.files,
                        chunk_size=req.chunk_size,
                        chunk_overlap=req.chunk_overlap)
        
        return IngestResponse(ok=True,
                              entitles_created=result["entitles_created"],
                              relationships_created=result["relationships_created"],
                              chunks_created=result["chunks_created"],
                              embedding_dim=result["embedding_dim"],
                              vector_index=result["vector_index"],
                              message="Ingestion completed")

    except Exception as e:
        raise HTTPException()
    finally:
        db.close()

@app.post("/query", response_model=QueryResponse)
def query(req: Queryrequest) -> QueryResponse:
    db = get_db()
    try:
        if not db.ping():
            raise HTTPException(status_code=503, detail="Neo4j not reachable.")
        
        if not db.vector_index_exists(index_name=settings.VECTOR_INDEX_NAME):
            raise HTTPException(status_code=400, detail="Vector index not found. Run /ingest first.")

        chuncks = vector_retrieve(neo4j=db, question=req.question, top_k=req.top_k)
        seed_chun_ids = [c.chunk_id for c in chuncks if c.chunk_id]

        paths = graph_expand(neo4j=db, seed_chunk_ids=seed_chun_ids, hops=req.hops) if req.hops > 0 else []
        
        context_pack = build_context_pack(chunks=chuncks, paths=paths)
        answer = answer_question(question=req.question, context_pack=context_pack)

        sources = [SourceItem(chunk_id=c.chunk_id, entity_id=c.entity_id, score=c.score) for c in chuncks]

        return QueryResponse(answer=answer,
                             sources=sources,
                             graph_paths=paths,
                             used_top_k=req.top_k,
                             used_hops=req.hops)

    except Exception:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed {e}")
    finally:
        db.close()