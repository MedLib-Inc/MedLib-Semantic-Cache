import chromadb
import logging
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Loads the embedding model."""
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, query: str) -> List[float]:
        """Generate an embedding for a given query."""
        return self.model.encode([query])[0]


class ChromaDatabase:
    def __init__(self, collection_name="query_cache", persist_path="./chromadb_storage", size=100):
        """Initializes the ChromaDB client and collection."""
        settings = Settings(allow_reset=True, persist_directory=persist_path)
        self.client = chromadb.PersistentClient(settings=settings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.size = size

    def add(self, query: str, response: str, embedding: List[float], timestamp: float) -> None:
        """Add query and response to the ChromaDB."""
        self.collection.add(
            documents=[response],
            metadatas=[{"query": query, "last_access": timestamp}],
            ids=[query],
            embeddings=[embedding]
        )
        logging.info(f"Added query '{query}' to the database. Cache size is now {self.count()}.")

    def get(self, query_embedding: List[float], top_k: int = 1) -> Optional[dict]:
        """Query ChromaDB for similar responses."""
        return self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, include=["documents", "distances"]
        )

    def count(self) -> int:
        """Return the number of entries in the cache."""
        return self.collection.count()

    def delete(self, query_ids: List[str]) -> None:
        """Delete entries from the cache."""
        self.collection.delete(ids=query_ids)

    def reset(self) -> None:
        """Reset the database and recreate the collection."""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
        logging.info("ChromaDB has been reset.")


class LRUCacheManager:
    def __init__(self, db: ChromaDatabase, size=100):
        """Initialize LRU cache manager."""
        self.db = db
        self.size = size

    def evict_if_needed(self) -> None:
        """Evict least recently used items if the cache size exceeds the limit."""
        if self.db.count() >= self.size:
            logging.info(f"Cache size {self.db.count()} exceeded limit {self.size}. Triggering eviction.")
            self.evict()

    def evict(self) -> None:
        """Evict the least recently used item."""
        results = self.db.collection.get(include=["metadatas"])
        items_with_time = sorted(
            [(item['id'], item['metadata']['last_access']) for item in results['metadatas']],
            key=lambda x: x[1]
        )
        num_to_evict = self.db.count() - self.size
        evict_ids = [item[0] for item in items_with_time[:num_to_evict]]
        self.db.delete(evict_ids)
        logging.info(f"Evicted {len(evict_ids)} items from cache.")
