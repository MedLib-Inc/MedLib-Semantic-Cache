import chromadb
from sentence_transformers import SentenceTransformer

class Chroma:
    def __init__(self, collection_name="query_cache"):
        """
        Initilizes the persistent cache
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
        """
        try:
            embedding = self.get_embedding(query)
            # Add document to ChromaDB
            self.collection.add(
                documents=[response],
                metadatas=[{"query": query}],
                ids=[query],
                embeddings=[embedding]
            )
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")

    def query_db(self, query, top_k=1):
        """
        Queries the ChromaDB collection for semantically similar responses
        """
        try:
            embedding = self.get_embedding(query)
            # Search based on embedding
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            if results['documents']:
                # Return top result
                return results['documents'][0]
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
        
        # If no result is found
        return None
    
    def clear_db(self):
        """
        Clears all the entries in the ChromaDB collection
        """
        self.collection.delete()
        #Recreate collection
        self.collection = self.client.get_or_create_collection("query_cache")