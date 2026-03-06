from __future__ import annotations

from neo4j import GraphDatabase
from typing import Any

from app.config import settings


class Neo4jClient:

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    
    def close(self) -> None:
        self._driver.close()
    

    def run_cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session as session:
            records = session.run(query, params or {})
            return [record.data() for record in records]
    

    def ping(self) -> bool:
        try:
            self.run_cypher("RETURN 1 AS ok")        
            return True
        except Exception:
            return False
    

    def ensure_constraints(self) -> None:
        self.run_cypher(            
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """
            )

        self.run_cypher(
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
        )
    
    def ensure_vector_index(self, index_name: str, dim: int) -> None:
        self.run_cypher(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """,
            {"dim": dim},
        )
    
    def vector_index_exists(self, index_name: str) -> bool:
        rows = self.run_cypher("SHOW INDEXES YIELD name WHERE name = $name RETURN name", {"name": index_name})
        return len(rows) > 0