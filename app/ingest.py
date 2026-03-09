from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
import json
import uuid
import os

from app.config import settings
from app.neo4j_client import Neo4jClient
from app.rag_chain import get_embeddings

@dataclass
class ParsedEntity:
    id: str
    label: str
    title: str
    text: str

@dataclass
class ParsedRelation:
    src_id: str
    rel_type: str
    dest_id: str



def _simple_chunk(text:str, chunk_size:int, overlap:int) -> list[str]:

    chunk: list[str] = []

    text = (text or "").strip()
    if not text:
        return chunk
    
    chunk_start = 0

    while chunk_start < len(text):
        chunk_end = min(len(text), chunk_start + chunk_size) # eg: min is 200
        chunk.append(text[chunk_start:chunk_end])

        if len(text) == chunk_end:
            break # If chunk is empty
        chunk_start = max(0, chunk_end - overlap) # chunk start gets updated.
    
    return chunk

def _load_csv(path: str) -> tuple[list[ParsedEntity], list[ParsedRelation]]:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    entities = list[ParsedEntity] = []
    relations = list[ParsedRelation] = []

    # ---- Mode A: record_type CSV (Entity / Rel) --
    if "record_type" in df.columns:
        for _, row in df.iterrows():
            rt = str((row.get("record_type") or "")).strip().upper()

            if rt == "ENTITY":
                entity_id = str(row.get("id") or "").strip()

                if not entity_id:
                    continue

                label = str(row.get("label") or "item").strip()
                title = str(row.get("title") or row.get("name" or entity_id)).strip()
                text = str(row.get("text") or row.get("description") or title).strip()
                entities.append(ParsedEntity(entity_id, label, title, text))

            elif rt == "REL":
                s = str(row.get("source_id") or "").strip()
                r = str(row.get("rel_type") or "").strip()
                d = str(row.get("target_id") or "").strip()
                if s and r and d:
                    relations.append(ParsedRelation(s,r,d))
            
        return entities, relations


    # ---- Mode B: simple CSV
    if "id" not in df.columns:
        return entities, relations

    for _, row in df.iterrows():
        entity_id = str(row.get("id") or "").strip()
        if not entity_id:
            continue

        label = str(row.get("label") or row.get("type") or "item")
        title = str(row.get("title") or row.get("name") or entity_id)
        text = str(row.get("text") or row.get("description") or title)
        entities.append(ParsedEntity(id=entity_id, label=label, title=title, text=text))
    
    rel_cols = {"rel_src", "rel_type", "rel_dst"}
    if rel_cols.issubset(set(df.columns)):
        for _, row in df.iterrows():
            s = str(row.get("rel_src"))
            t = str(row.get("rel_type"))
            d = str(row.get("rel_dst"))
            if pd.notna(s) and pd.notna(t) and pd.notna(d):
                relations.append(ParsedRelation(src_id=s, rel_type=t, dest_id=d))
    
    return entities, relations


def _load_json(path: str) -> tuple[list[ParsedEntity], list[ParsedRelation]]:
    
    with open(file=path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    entities: list[ParsedEntity] = []
    relations: list[ParsedRelation] = []

    if isinstance(data, list):
        for obj in data:
            entity_id = str((obj.get("id") or obj.get("entity_id") or uuid.uuid4().hex))
            label = str(obj.get("label") or obj.get("type") or "item")
            title = str(obj.get("title") or obj.get("name") or entity_id)
            text = str(obj.get("text") or obj.get("description") or title)
            entities.append(ParsedEntity(entity_id, label, title, text))

            return entities, relations
    
    if isinstance(data, dict):
        for obj in data.get("entities", []):
            entity_id = str(obj.get("id") or obj.get("entity_id") or uuid.uuid4().hex)
            lable = str(obj.get("lable") or obj.get("type") or "item")
            title = str(obj.get("title") or obj.get("name") or entity_id)
            text = str(obj.get("text") or obj.get("description") or title)
            entities.append(ParsedEntity(entity_id, lable, title, text))
        
        for obj in data.get["relationships", []]:
            s = str(obj.get("src_id") or obj.get("source") or obj.get("source_id") or "")
            t = str(obj.get("rel_type") or obj.get("type") or "")
            d = str(obj.get("dst_id") or obj.get("target") or obj.get("target_id") or "")
            if pd.notna(s) and pd.notna(t) and pd.notna(d):
                relations.append(ParsedRelation(s,t,d))
        
        return entities, relations
    return entities, relations

def _load_text(path: str) -> tuple[list[ParsedEntity], list[ParsedRelation]]:

    with open(path,"r", encoding="utf-8") as f:
        text = f.read()
    base = os.path.basename(path)
    entity_id = os.path.splitext(base)[0]
    return [ParsedEntity(entity_id,"Document",base,text)], []

def load_data_files(data_dir: str, files: list[str] | None) -> tuple[list[ParsedEntity], list[ParsedRelation]]:
    entities: list[ParsedEntity] = []
    relations: list[ParsedRelation] = []

    if files:
        paths = [os.path.join(data_dir, f) for f in files]
    else:
        paths = []
        for name in os.listdir(data_dir):
            if name.lower().endswith((".csv",".json", ".txt")):
                paths.append(os.path.join(data_dir, name))
    
    for path in paths:
        if not os.path.exists(path):
            continue

        if path.lower().endswith(".csv"):
            e,r = _load_csv(path)
        elif path.lower().endswith(".json"):
            e,r = _load_json(path)
        elif path.lower().endswith(".txt"):
            e, r = _load_text(path)
        else:
            continue
        entities.extend(e)
        relations.extend(r)
    
    dedup: dict[str, ParsedEntity] = {e.id: e for e in entities}

    return list(dedup.values()), relations

def ingest_to_neo4j(
    neo4j: Neo4jClient,
    files: list[str] | None,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    neo4j.ensure_constraints()

    entities, relations = load_data_files(settings.DATA_DIR, files)

    # 1) Upsert Entities
    for e in entities:
        neo4j.run_cypher(
            """
            MERGE (n:Entity {id: $id})
            SET n.label = $label,
                n.title = $title,
                n.text = $text
            """,
            {"id": e.id, "label": e.label, "title": e.title, "text": e.text},
        )

    # 2) Upsert relationships (optional)
    # We store relationship type in property rel_type to keep MVP safe (dynamic REL types are possible but more complex)
    rel_created = 0
    for r in relations:
        neo4j.run_cypher(
            """
            MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})
            MERGE (a)-[rel:RELATED_TO {rel_type: $type}]->(b)
            """,
            {"src": r.src_id, "dst": r.dst_id, "type": r.rel_type},
        )
        rel_created += 1

    # 3) Create Chunk nodes, embeddings, HAS_CHUNK relationships
    embedder = get_embeddings()
    embedding_dim: int | None = None
    chunks_created = 0

    for e in entities:
        chunks = _simple_chunk(e.text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            continue

        vectors = embedder.embed_documents(chunks)  # list[list[float]]
        if vectors and embedding_dim is None:
            embedding_dim = len(vectors[0])

        for chunk_text, vec in zip(chunks, vectors):
            chunk_id = uuid.uuid4().hex
            neo4j.run_cypher(
                """
                MATCH (e:Entity {id: $entity_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.embedding = $embedding
                MERGE (e)-[:HAS_CHUNK]->(c)
                """,
                {
                    "entity_id": e.id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "embedding": vec,
                },
            )
            chunks_created += 1

    if embedding_dim is None:
        # No text found / no embeddings created
        embedding_dim = 1536  # safe default; index will exist even if empty

    # 4) Ensure vector index exists
    neo4j.ensure_vector_index(settings.VECTOR_INDEX_NAME, embedding_dim)

    # Counts (rough, MVP)
    entity_count = neo4j.run_cypher("MATCH (e:Entity) RETURN count(e) AS c")[0]["c"]
    chunk_count = neo4j.run_cypher("MATCH (c:Chunk) RETURN count(c) AS c")[0]["c"]

    return {
        "entities_created": int(entity_count),
        "relationships_created": int(rel_created),
        "chunks_created": int(chunk_count),
        "embedding_dim": int(embedding_dim),
        "vector_index": settings.VECTOR_INDEX_NAME,
    }