# semantic_cache.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def _init__(self, threshold=0.9):
        """
        Initializes the semantic cache with an embedding model and a similarity threshold
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.cache = [] #temporarily using in-memory list to store (query, embedding, response) tuples

def get_embedding(self, query):
    """
    Convert a query to its embedding using SentenceTransformers
    """
    return self.model.encode([query])[0]

def add_to_cache(self, query, response):
    """
    Add query and response to cache
    """

    embedding = self.get_embedding(query)
    # add embedding to cache (ChromaDB)
    self.cache.append((query, embedding, response)) # temp in-memory cache

def check_cache(self, query):
    """
    Check if a semantically similar response is in the cache
    """
    
    embedding = self.get_embedding(query)
    # implement with database
    # search cache for closest match
    # if match is within threshold, return
    # else, return None

    # iterate over cached items to find match
    for cached_query, cached_embedding, cached_response in self.cache:
        similarity = cosine_similarity([embedding], [cached_embedding])[0][0]
        if similarity >= self.treshold:
            return cached_response
    
    # if no similar query found
    return None

def ask(self, query):
    """
    Main method to handle a query: check cache or return a new response
    """
    
    cached_response = self.check_cache(query)
    # implement with database
    # if cached reponse exists, return it
    # else, make new query to LLM, add it to the cache, and return response

    if cached_response:
        return cached_response
    
    # if not found, return generic response
    response = self.query_external(query)

    self.add_to_query(query, response)
    return response

def query_external(self, query):
    """
    Call the LLM for a new response is one is not found in the cache
    """
    # implement actual LLM logic, return generated reponse
    return f"Generated response for: {query}"
