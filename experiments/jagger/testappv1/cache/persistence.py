# persistence.py
import chromadb
from chromadb.config import Settings

class Persistence:
    def __init__(self):
        #self.client = chromadb.Client(chroma_db_impl="duckdb+parquet", persist_directory="chroma_storage")
        #self.client = chromadb.Client(persist_directory="chroma_storage")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("responses")
    
    def add_response(self, query, response, embedding):
        self.collection.add(document=[response], embeddings=[embedding], metadatas=[{"query": query}])

    def get_response(self, query):
        return "Response from persistence"

