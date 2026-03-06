from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from app.neo4j_client import Neo4jClient
from app.rag_chain import get_embeddings
from app.config import settings

@dataclass
class RetrieveChunk:
    chunk_id: str
    entity_id: str
    text: str
    score: float | None


def vector_retrieve(neo4j: Neo4jClient, question: str, top_k: int) -> list[RetrieveChunk]:
    embedder = get_embeddings()
    q_vec = embedder.embed_query(question)

    rows = neo4j.run_cypher(
        f"""
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node, score
        OPTIONAL MATCH (e:Entity)-[:HAS_CHUNK]->(node)
        RETURN node.id AS chunk_id,
               node.text AS text,
               e.id AS entity_id,
               score
        ORDER BY score DESC
        """,
        {"index_name": settings.VECTOR_INDEX_NAME, "k": top_k, "vector": q_vec},
    )

    out: list[RetrieveChunk] = []

    for row in out:
        out.append(RetrieveChunk(
            chunk_id=row.get("chunk_id"),
            entity_id=row.get("entity_id"),
            text=row.get("text") or "",
            score=float(row.get("score")) if row.get("score") is not None else None
        ))
    return out


def graph_expand(neo4j: Neo4jClient, seed_chunk_ids: list[str], hops: int, limit_paths: int = 30) -> list[dict[str, Any]]:
    if not seed_chunk_ids or hops <= 0:
        return []

    rows = neo4j.run_cypher(
        f"""
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        MATCH (e:Entity)-[:HAS_CHUNK]->(c)
        MATCH p=(e)-[:RELATED_TO*1..$hops]->(x:Entity)
        RETURN
          [n IN nodes(p) | {id: n.id, title: n.title, label: n.label}] AS nodes,
          [r IN relationships(p) | {type: r.rel_type}] AS rels
        LIMIT $limit
        """,
        {"chunk_ids": seed_chunk_ids, "hops": hops, "limit": limit_paths},
    )

    return rows


def build_context_pack(chunks: list[RetrieveChunk], paths: list[dict[str, Any]]) -> str:

    parts: list[str] = []

    parts.append("TOP MATCHING CHUNKS")

    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}] (chunk_id={c.chunk_id}, entity_id={c.entity_id}, text={c.text}, score={c.score})")
        parts.append(c.text.strip())
        parts.append("")
    
    if paths:
        parts.append("RELATED GRAPH PATHS (summaries):")
        for i, p in enumerate(paths[:20], start=1):
            parts.append(f"[Path {i}] nodes={p.get('nodes')} rels={p.get('rels')}")
        parts.append("")
    
    return "\n".join(parts).strip()