# endpoints.py
from fastapi import APIRouter, HTTPException
from ..models.models import QueryRequest, QueryResponse
from ..cache.semantic_cache import SemanticCache

router = APIRouter()
semantic_cache = SemanticCache()

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        response = semantic_cache.get_response(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


