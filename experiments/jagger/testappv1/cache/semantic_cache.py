# semantic_cache.py
from sentence_transformers import SentenceTransformer
from testappv1.cache.persistence import store_embedding, search_similar_queries

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_and_store_embedding(query: str):
    embedding = model.encode(query).tolist()
    store_result = store_embedding(query, embedding)
    return embedding, store_result

def get_similar_queries(embedding: list):
    search_results = search_similar_queries(embedding)
    similar_queries = [
        {"query": doc, "score": score}
        for doc, score in zip(search_results['documents'], search_results['distances'])
    ]
    return similar_queries
