import chromadb
import logging

from sentence_transformers import SentenceTransformer

class Chroma:
    def __init__(self, collection_name="query_cache", persist_path="./chromadb_storage"):
        """
        Initilizes the persistent cache
        """

        # Persistent ChromaDB client, stores data in path
        self.client = chromadb.PersistentClient(path=persist_path)

        # Create/load collection
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        logging.basicConfig(level=logging.INFO)

    def get_embedding(self, query):
        """
        Convert a query to its embedding using SentenceTransformers
        """
        return self.model.encode([query])[0]
    
    def add_to_db(self, query, response):
        """
        Adds a query, embedding, and response to the ChromaDB collection.
        """
        try:
            embedding = self.get_embedding(query)
            logging.info(f"Adding query: {query} with embedding: {embedding}")

            # Add document to ChromaDB
            self.collection.add(
                documents=[response],
                metadatas=[{"query": query}],
                ids=[query],
                embeddings=[embedding]
            )
        except Exception as e:
            logging.error(f"Error adding to ChromaDB: {e}")

    def query_db(self, query, top_k=1):
        """
        Queries the ChromaDB collection for semantically similar responses
        """
        try:
            embedding = self.get_embedding(query)
            logging.info(f"Querying ChromaDB with embedding: {embedding}")

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )

            logging.info(f"Query results: {results}")

            if results['documents']:
                return results['documents'][0]
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")

        return None
    
    def clear_db(self):
        """
        Clears all the entries in the ChromaDB collection
        """
        try:
            self.collection.delete()
            logging.info("ChromaDB collection cleared.")

            # Recreate the collection
            self.collection = self.client.get_or_create_collection("query_cache")
        except Exception as e:
            logging.error(f"Error clearing ChromaDB: {e}")
            # Propagate error to the API
            raise e