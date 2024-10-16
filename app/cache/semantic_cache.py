from .persistence import Chroma

class SemanticCache:
    def __init__(self, threshold=0.9):
        """
        Initializes the semantic cache with ChromaDB and a similarity threshold
        
        Usage in API:
        -------------
        Initalize the cache in main FastAPI app or router:
        
        from ..cache.semantic_cache import SemanticCache

        semantic_cache = SemanticCache()
        """
        self.persistence = Chroma()
        self.threshold = threshold

    def add_to_cache(self, query, response):
        """
        Add query and response to cache (temporarily in memory).
        Later to be swapped out for a persistent database.

        Usage in API:
        -------------
        You can use this method in the '/queries/add' endpoint to add query-response pair

        @router.post("/queries/add")
           async def add_query_response(data: QueryResponse):
               query = data.query
               response = data.response
               semantic_cache.add_to_cache(query, response)
               # Continue with your logic to store in-memory or persist to file.
        """
        # Store in ChromaDB
        self.persistence.add_to_db(query, response)

    def check_cache(self, query):
        """
        Check if a semantically similar response is in the cache.
        """
        # Query ChromaDB for similar response
        result = self.persistence.query_db(query)

        if result:
            return result # Return cached response if found

        # Return None if no match is found
        return None

    def ask(self, query):
        """
        Main method to handle a query: check cache or return a new response.
        Returns cached response if available, otherwise queries LLM (simulated for now).
        
        Usage in API:
        -------------
        Call this method in the '/queries/get/{query}' endpoint to handle a query

        @router.get("/queries/get/{query}")
        async def get_diagnosis(query: str):
            response = semantic_cache.ask(query)
            return {"query": query, "diagnosis": response}
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
