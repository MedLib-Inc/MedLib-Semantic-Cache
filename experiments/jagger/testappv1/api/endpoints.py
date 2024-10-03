# endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from testappv1.cache.semantic_cache import generate_and_store_embedding, get_similar_queries

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint is working"}

class QueryRequest(BaseModel):
    query: str

@router.post("/generate_embedding")
async def generate_embedding(request: QueryRequest):
    embedding, store_result = generate_and_store_embedding(request.query)
    similar_queries = get_similar_queries(embedding)
    return {
        "query": request.query,
        "similar_queries": similar_queries
    }
