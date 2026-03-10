# graphRag
This is a project of GraphRAG, where you search semi-structured (vectors) and structured (knowledge graph/neo4j).

# Plan
Short & Simple MVP plan (Neo4j + LangChain + FastAPI + sample data) that you can build in small steps.

### 2) Porposed project structure
## Project Structure

```text
app/
  main.py                # FastAPI endpoints (API routes)
  config.py              # Configuration + environment variables (Neo4j URL/user/password, model, etc.)
  neo4j_client.py        # Neo4j driver wrapper + helper methods (e.g., run_cypher())
  ingest.py              # Ingestion pipeline: load sample data → build nodes/edges → chunk text → create embeddings → store vector index
  retriever.py           # Hybrid retrieval: vector top-k search + graph expansion (Neo4j traversal)
  rag_chain.py           # RAG pipeline: build prompt → call LLM → return answer + sources
  schemas.py             # Pydantic request/response models (API contracts)

data/
  sample.csv             # Sample structured dataset (CSV)
  sample.json            # Sample structured dataset (JSON)
  sample.txt             # Sample unstructured dataset (text)
```  

## Start the services
### Start Neo4j
docker start neo4j-rag
docker ps | grep neo4j-rag

### Neo4j url
http://localhost:7474

## Launch unicorn
uvicorn app.main:app --reload --port 8000

## Fast API url:
http://127.0.0.1:8000/docs


### Questions you could ask
1. Which movies did Keanu Reeves act in?
2. What is the Matrix about?
3. Which movies are set in Los Angeles?
4. Tell me movies connected to Artificial Reality and who acted in them.