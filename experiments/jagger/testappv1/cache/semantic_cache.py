# semantic_cache.py
from sentence_transformers import SentenceTransformer
from .persistence import Persistence

class SemanticCache:
    def __init__(self):
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.persistence = Persistence()

    def get_response(self, query):
        embedding = self.model.encode(query)

        response = "Simulated response for query:" + query
        self.persistence.add_response(query, response, embedding)
        return response
    
    
