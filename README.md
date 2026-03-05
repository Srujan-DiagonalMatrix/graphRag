# graphRag
This is a project of GraphRAG, where you search semi-structured (vectors) and structured (knowledge graph/neo4j).

# Plan
Short & Simple MVP plan (Neo4j + LangChain + FastAPI + sample data) that you can build in small steps.

# 1) MVP architecture (what you’re building)

Neo4j stores:
Graph = entities + relationships (structured)
Vector index = embeddings for text chunks (unstructured)
LangChain does: embedding + retrieval (vector + Cypher) + LLM answering
FastAPI exposes: /ingest, /query, /health

# 2) Suggested project structure
app/
  main.py                # FastAPI routes
  config.py              # env vars (neo4j url/user/pwd, model, etc.)
  neo4j_client.py        # driver + helper run_cypher()
  ingest.py              # load sample data -> nodes/edges -> chunks -> embeddings -> vector index
  retriever.py           # hybrid retrieval (vector top-k + graph expansion)
  rag_chain.py           # build prompt + call LLM + return answer + sources
  schemas.py             # Pydantic request/response models
data/
  sample.csv / sample.json / sample.txt

# 3) Checklist plan (build in this order)
# Step 1 — Setup & connectivity
Run Neo4j (Docker) and confirm Neo4j Browser works.
Create constraints/index basics (unique id, labels).
Add FastAPI skeleton with /health that pings Neo4j.

# Step 2 — Load sample dataset into a Knowledge Graph
Pick a simple dataset (easy MVP options):
Movies (people–acted_in–movies), or
Products (product–belongs_to–category), or
Org chart (employee–reports_to–manager)

# Ingest pipeline:
Parse file → create nodes + relationships in Neo4j.
Create a text field per entity (e.g., “Movie plot + cast”, “Product description”).

# Step 3 — Add unstructured chunks + vector index (inside Neo4j)
Split entity text into chunks (e.g., 300–600 tokens).
Create :Chunk nodes connected to entities (e.g., (:Movie)-[:HAS_CHUNK]->(:Chunk)).
Generate embeddings using LangChain → store embedding vector on each :Chunk.
Create Neo4j vector index on :Chunk.embedding.

# Step 4 — Hybrid retrieval (GraphRAG)
For a user question:
Vector retrieve: top-k :Chunk by similarity.
Graph expand: from the chunks → go to linked entities → fetch related nodes (1–2 hops) via Cypher.
Combine into a “context pack”: {top_chunks, related_entities, relationships}.

# Step 5 — RAG answer + FastAPI endpoint
Build a LangChain chain: context_pack + question -> LLM -> answer.

# FastAPI /query returns:
answer
sources (chunk ids / entity ids)
graph_paths (optional: the nodes/relationships used)

# Step 6 — Minimal quality + demo
Add 5–10 test questions (both structured + unstructured):
“Who acted in X?” (graph)
“What is X about?” (vector)
“Which movies similar to X involve actor Y?” (hybrid)
Add simple logging + timing in the response.

# 4) FastAPI endpoints (MVP)
POST /ingest → loads sample data, builds graph, chunks, embeddings, vector index
POST /query → {question, top_k=5, hops=1} → returns answer + sources
GET /health → checks Neo4j connection + index exists
