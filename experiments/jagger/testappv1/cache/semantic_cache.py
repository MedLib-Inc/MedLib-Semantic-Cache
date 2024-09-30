# semantic_cache.py
from sentence_transformers import SentenceTransformer
from testappv1.cache.persistence import get_or_create_collection

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB collection
collection_name = "dummy_collection"
collection = get_or_create_collection(collection_name)

# Dummy function
async def handle_query(query: str):
    return {
        "query": query,
        "result": "This is a dummy response.",
        "cached": False,
    }