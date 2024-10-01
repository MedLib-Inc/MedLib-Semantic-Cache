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


