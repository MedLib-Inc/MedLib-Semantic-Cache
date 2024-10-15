from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.9):
        """
        Initializes the semantic cache with an embedding model and a similarity threshold
        
        Usage in API:
        -------------
        Initalize the cache in main FastAPI app or router:
        
        from ..cache.semantic_cache import SemanticCache

        semantic_cache = SemanticCache()
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.cache = []  # Temporarily using in-memory list to store (query, embedding, response) tuples

    def get_embedding(self, query):
        """
        Convert a query to its embedding using SentenceTransformers
        """
        return self.model.encode([query])[0]

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
        embedding = self.get_embedding(query)
        self.cache.append((query, embedding, response))  # Temporary in-memory cache

    def check_cache(self, query):
        """
        Check if a semantically similar response is in the cache.
        """
        embedding = self.get_embedding(query)

        # Iterate over cached items to find a match
        for cached_query, cached_embedding, cached_response in self.cache:
            #similarity = cosine_similarity([embedding], [cached_embedding])[0][0]
            if similarity >= self.threshold:
                return cached_response

        # If no similar query is found
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
            return cached_response

        # If not found in the cache, simulate a call to the LLM (or external system)
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
