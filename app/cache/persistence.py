import chromadb
from sentence_transformers import SentenceTransformer

class Chroma:
    def __init__(self, collection_name="query_cache"):
        """
        Initilizes the persistent cache

        Import the Chroma class into the `semantic_cache.py` file:
        from .persistence import Chroma
        There is also now no need to import sentence_transformers or sklearn
        since ChromaDB can handle that
        
        In sematic_cache.py, `__init__()`:
        ----------------
        Replace: self.cache = []
        With: self.persistence = ChromaPersistence()
        """

        # ChromaDB client
        self.client = chromadb.Client()

        # Create/load collection
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embedding(self, query):
        """
        Convert a query to its embedding using SentenceTransformers
        """
        return self.model.encode([query])[0]
    
    def add_to_db(self, query, response):
        """
        Adds a query, embedding, and response to the ChromaDB collection
        
        In semantic_cache.py, 'add_to_cache():
        ---------------------
        Replace: self.cache.append((query, embedding, response))
        With: self.persistence.add_to_db(query, response)
        """
        embedding = self.get_embedding(query)

        # Add document to ChromaDB
        self.collection.add(
            documents=[response],
            metadatas=[{"query": query}],
            ids=[query],
            embeddings=[embedding]
        )

    def query_db(self, query, top_k=1):
        """
        Queries the ChromaDB collection for semantically similar responses
        
        In semantic_cache.py, 'check_cache()':
        ---------------------
        Replace the in-memory cosine similarity check with:
          result = self.persistence.query_db(query)
          if result:
              return result
        """

        embedding = self.get_embedding(query)

        # Search based on embedding
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        if results['documents']:
            # Return top result
            return results['documents'][0]
        
        # If no result is found
        return None
    
    def clear_db(self):
        """
        Clears all the entries in the ChromaDB collection
        """
        self.collection.delete()
        #Recreate collection
        self.collection = self.client.get_or_create_collection("query_cache")