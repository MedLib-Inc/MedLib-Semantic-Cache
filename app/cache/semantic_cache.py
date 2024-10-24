import logging
import time
import requests
import os
from .persistence import ChromaDatabase, EmbeddingService, LRUCacheManager
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api-inference.huggingface.co/models/google/flan-t5-base")

# https://api-inference.huggingface.co/models/google/flan-t5-small
# https://api-inference.huggingface.co/models/google/flan-t5-base
# https://api-inference.huggingface.co/models/google/flan-t5-large

if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

class SemanticCache:
    def __init__(self, db: ChromaDatabase, cache_manager: LRUCacheManager, threshold:float = 0.15):
        """Initialize the semantic cache with ChromaDB, LRU manager, and similarity threshold."""
        self.db = db
        self.cache_manager = cache_manager
        self.threshold = threshold
        self.embedding_service = EmbeddingService()

    def add_to_cache(self, query: str, response: str) -> None:
        """Add a query and response to the cache."""
        embedding = self.embedding_service.generate_embedding(query)
        timestamp = time.time()
        self.cache_manager.evict_if_needed()
        self.db.add(query, response, embedding, timestamp)

    def remove_from_cache(self, query: str) -> None:
        """Remove a query from the cache."""
        try:
            self.db.delete([query])
            logging.info(f"Removed query '{query}' from the cache.")
        except Exception as e:
            logging.error(f"Error removing query '{query}': {e}")

    def check_cache(self, query: str) -> Optional[str]:
        """Check if a semantically similar response is in the cache."""
        embedding = self.embedding_service.generate_embedding(query)
        results = self.db.get(embedding)

        if not results or 'distances' not in results or not results['distances']:
            logging.warning(f"No valid results found for query: '{query}'")
            return None

        top_distance = results['distances'][0][0]
        if top_distance <= self.threshold:
            logging.info(f"Cache hit for query: '{query}' with response: {results['documents'][0]}")
            return results['documents'][0]
        else:
            logging.info(f"Cache miss: Distance {top_distance} exceeds threshold.")
        return None

    def ask(self, query: str) -> str:
        """Handle a query: return cached response if available, otherwise query the LLM."""
        cached_response = self.check_cache(query)
        if cached_response:
            return cached_response

        # Cache miss: Query the LLM
        response = self.query_external(query)
        self.add_to_cache(query, response)
        return response

    def query_external(self, query: str) -> str:
        """Call the LLM API for a response if not found in the cache."""
        logging.info(f"Calling LLM for query: '{query}'")
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        payload = {"inputs": query}

        try:
            response = requests.post(LLM_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if result and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
                logging.info(f"LLM response: '{generated_text}' for query: '{query}'")
                return generated_text
            else:
                logging.warning(f"No valid response from LLM for query: '{query}'")
                return f"No valid response generated for: {query}"
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return f"Error generating response for: {query}"

    def update_threshold(self, new_threshold: float) -> None:
        """Update the similarity threshold."""
        self.threshold = new_threshold
        logging.info(f"Threshold updated to {new_threshold}.")

    def update_cache_size(self, new_size: int) -> None:
        """Update the cache size and evict if necessary."""
        self.cache_manager.size = new_size
        self.cache_manager.evict_if_needed()
        logging.info(f"Cache size updated to {new_size}.")

