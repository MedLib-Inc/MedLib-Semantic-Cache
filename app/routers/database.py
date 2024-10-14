# API Endpoints

from fastapi import APIRouter
from pydantic import BaseModel
from ..cache.persistence import Chroma 
from ..utility.response_formatter import create_response 

router = APIRouter()

chroma_persistence = Chroma()

class QueryResponse(BaseModel):
    query: str
    response: str

@router.get("/database/get/{query}")
async def get_database(query: str):
    result = chroma_persistence.query_db(query)
    if result:
        return create_response(status="success", data={"query": query, "diagnosis": result})
    
    return create_response(status="error", message="No diagnosis found for this query.")

@router.post("/database/add")
async def add_database(data: QueryResponse):
    query = data.query
    response = data.response
    
    try:
        chroma_persistence.add_to_db(query, response)
    except Exception as e:
        return create_response(status="error", message="Failed to add to ChromaDB.")
    
    return create_response(status="success", message="Query-response pair added successfully.")

@router.post("/database/clear")
async def clear_cache():
    try:
        chroma_persistence.clear_db()
    except Exception as e:
        return create_response(status="error", message="Failed to clear ChromaDB.")
    
    return create_response(status="success", message="Cache cleared successfully.")

@router.get("/database")
async def get_root():
    return create_response(status="success", data={"message": "Hello, Queries!"})
