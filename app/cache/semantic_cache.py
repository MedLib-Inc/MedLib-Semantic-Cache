# Semantic Cache main file
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def _init__(self, threshold=0.9):
        """
        Initializes the semantic cache with an embedding model and a similarity threshold
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

def get_embedding(self, query):
    """
    Convert a query to its embedding using SentenceTransformers
    """
    return self.model.encode(query)

def add_to_cache(self, query, response):
    """
    Add query and response to cache
    """

    embedding = self.get_embedding(query)
    # add embedding to cache (ChromaDB)

def check_cache(self, query):
    """
    Check if a semantically similar response is in the cache
    """
    
    embedding = self.get_embedding(query)
    # search cache for closest match
    # if match is within threshold, return
    # else, return None

def ask(self, query):
    """
    Main method to handle a query: check cache or return a new response
    """
    
    cached_response = self.check_cache(query)
    # if cached reponse exists, return it
    # else, make new query to LLM, add it to the cache, and return response

def query_external(self, query):
    """
    Call the LLM for a new response is one is not found in the cache
    """
    # return generated reponse
