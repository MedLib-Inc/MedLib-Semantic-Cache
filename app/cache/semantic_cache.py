import logging
from .persistence import Chroma

class SemanticCache:
    def __init__(self, threshold=0.1):
        """
        Initializes the semantic cache with ChromaDB and a similarity threshold
        """
        self.persistence = Chroma(threshold=threshold) # Pass threshold to ChromaDB client

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'  # Add timestamps
        )
        logging.info("SemanticCache initialized.")
        
    def add_to_cache(self, query, response):
        """
        Add query and response to the persistent cache (ChromaDB).
        """
        # Store in ChromaDB
        self.persistence.add_to_db(query, response)

    def check_cache(self, query):
        """
        Check if a semantically similar response is in the cache.
        """
        # Query ChromaDB for similar response
        results = self.persistence.query_db(query)
    
        if results and results['documents']:
            top_distance = results['distances'][0][0]
            logging.info(f"Top result distance for '{query}': {top_distance} (Threshold: {self.persistence.threshold})")

            # Apply threshold comparison (1 - cosine similarity), smaller is more similar
            if top_distance <= self.persistence.threshold:
                logging.info(f"Cache hit for query: '{query}'")
                return results['documents'][0]
            else:
                logging.info(f"Cache miss: Distance {top_distance} exceeds threshold.")
        return None

    def ask(self, query):
        """
        Main method to handle a query: check cache or return a new response.
        Returns cached response if available, otherwise queries LLM (simulated for now).
        """
        cached_response = self.check_cache(query)

        if cached_response:
            # Return cached response if found
            return cached_response

        # If not found in the cache, simulate a calling the LLM
        response = self.query_external(query)

        # Add the new response to the cache
        self.add_to_cache(query, response)

        return response

    def query_external(self, query):
        """
        Call the LLM for a new response if one is not found in the cache.
        Replace this with actual LLM logic in the future.
        """
        return f"Generated response for: {query}"
