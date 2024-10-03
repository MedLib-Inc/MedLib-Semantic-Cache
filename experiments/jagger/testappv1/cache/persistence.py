# persistence.py
import chromadb
import uuid

client = chromadb.PersistentClient(path="chroma_storage")

collection = client.get_or_create_collection("query_embeddings")

def store_embedding(query: str, embedding: list):

    unique_id = str(uuid.uuid4())

    collection.add(
        ids=[unique_id],
        documents=[query],
        embeddings=[embedding]
    )
    return {"status": "embedding stored", "id":unique_id}

def search_similar_queries(embedding: list, n_results: int = 5):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results

