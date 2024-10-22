import time
import chromadb
import logging
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class Chroma:
    def __init__(self, collection_name="query_cache", persist_path="./chromadb_storage", size=100, threshold=0.1):
        """
        Initializes the persistent cache with a ChromaDB collection using cosine similarity.
        """

        # Use the Settings object to allow reset and set persist directory
        settings = Settings(
            allow_reset=True,
            persist_directory=persist_path,
        )

        self.size = size
        self._persist_directory = settings.persist_directory

        # Persistent ChromaDB client with configured settings
        self.client = chromadb.PersistentClient(settings=settings)

        # Create/load collection with cosine similarity as the distance function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Ensures cosine similarity is used
        )

        # Load the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Similarity threshold for accepting a cached result
        self.threshold = threshold

        # Set up logging globally in the entry point
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("ChromaDB initialized.")

    def get_embedding(self, query):
        """
        Convert a query to its embedding using SentenceTransformers.
        """
        embedding = self.model.encode([query])[0]
        logging.info(f"Generated embedding: {embedding[:3]}... for query: '{query}'")
        return embedding

    def add_to_db(self, query, response):
        """
        Adds a query, embedding, and response to the ChromaDB collection.
        """
        try:
            embedding = self.get_embedding(query)
            current_time = time.time()
            logging.info(f"Adding query: '{query}' to the database.")

            if self.collection.count() >= self.size:
                self.lru_evict()

            # Add document to ChromaDB
            self.collection.add(
                documents=[response],
                metadatas=[{"query": query, "last_access": current_time}],
                ids=[query],
                embeddings=[embedding]
            )
            logging.info(f"Successfully added query: '{query}' to the database.")
        except Exception as e:
            logging.error(f"Error adding query '{query}' to ChromaDB: {e}")

    def lru_evict(self):
        count = self.collection.count()
        logging.info(f"Cache size exceeded. Current size: {count}, Max size: {self.size}")

        # Fetch all entries and sort by the 'last_access' timestamp
        results = self.collection.get(include=["metadatas"])
        items_with_time = [
            (item['id'], item['metadata']['last_access'])
            for item in results['metadatas']
        ]
        # Sort by access time (ascending order, least recently used first)
        items_with_time.sort(key=lambda x: x[1])

        # Evict the oldest items
        num_to_evict = count - self.size
        evict_ids = [item[0] for item in items_with_time[:num_to_evict]]
        self.collection.delete(ids=evict_ids)
        logging.info(f"Evicted {len(evict_ids)} items from cache.")

    def remove_from_db(self, query):
        """
        Remove query and its respective attributes by its query id
        """
        try:
            logging.info(f"Removing query: '{query}' to the database.")

            self.collection.delete(ids=[query])

            logging.info(f"Successfully removed query: '{query}' to the database.")
        except Exception as e:
            logging.error(f"Error removed query '{query}' to ChromaDB: {e}")

    def query_db(self, query, top_k=1):
        """
        Queries the ChromaDB collection for semantically similar responses.
        """
        try:
            embedding = self.get_embedding(query)
            logging.info(f"Searching for similar responses for query: '{query}'")

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "distances"]
            )
            logging.info(f"Query results: {results}")

            return results
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            return None

    def clear_db(self):
        """
        Clears the ChromaDB collection by resetting the database and recreating the collection.
        """
        try:
            # Reset the entire client (this deletes all collections)
            self.client.reset()
            logging.info("ChromaDB cache reset successfully.")

            # Recreate the collection after reset
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logging.info("Collection recreated after reset.")
        except Exception as e:
            logging.error(f"Error resetting ChromaDB: {e}")
            raise e  # Ensure the exception propagates to the API

    def configure_threshold(self, threshold):
        self.threshold = threshold

    def change_size(self, size):
        self.size = size
        if self.collection.count() >= self.size:
            self.lru_evict()
