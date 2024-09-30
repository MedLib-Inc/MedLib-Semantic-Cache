# endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from testappv1.cache.semantic_cache import handle_query

router = APIRouter()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    cached: bool

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    # Call semantic cache logic
    result = await handle_query(request.query)
    return QueryResponse(response=result['result'], cached=result['cached'])
