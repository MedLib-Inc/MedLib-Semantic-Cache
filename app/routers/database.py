# API Endpoints

from fastapi import APIRouter
from pydantic import BaseModel
from ..cache.semantic_cache import SemanticCache
from ..utility.response_formatter import create_response 

router = APIRouter()

semantic_cache = SemanticCache()

class QueryResponse(BaseModel):
    query: str
    response: str

@router.get("/database/get/{query}")
async def get_database(query: str):
    result = semantic_cache.ask(query)
    if result:
        return create_response(status="success", data={"query": query, "diagnosis": result})
    
    return create_response(status="error", message="No diagnosis found for this query.")

@router.post("/database/add")
async def add_database(data: QueryResponse):
    query = data.query
    response = data.response
    
    try:
        semantic_cache.add_to_cache(query, response)
    except Exception as e:
        return create_response(status="error", message="Failed to add to ChromaDB.")
    
    return create_response(status="success", message="Query-response pair added successfully.")

@router.post("/database/clear")
async def clear_cache():
    try:
        semantic_cache.persistence.clear_db()
    except Exception as e:
        return create_response(status="error", message="Failed to clear ChromaDB.")
    
    return create_response(status="success", message="Cache cleared successfully.")

@router.get("/database")
async def get_root():
    return create_response(status="success", data={"message": "Hello, Queries!"})
