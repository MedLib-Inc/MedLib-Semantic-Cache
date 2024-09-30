# persistence.py
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def get_or_create_collection(collection_name: str):
    return {"collection_name": collection_name, "status": "dummy_collection"}
