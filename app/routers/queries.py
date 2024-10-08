# API Endpoints

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.data import medical_queries_responses
import os

router = APIRouter()

DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "../models/data.py")

class QueryResponse(BaseModel):
    query: str
    response: str

@router.get("/queries/get/{query}")
async def get_diagnosis(query: str):
    response = medical_queries_responses.get(query, "No diagnosis found for this query.")
    return {"query": query, "diagnosis": response}

@router.post("/queries/add")
async def add_query_response(data: QueryResponse):
    query = data.query
    response = data.response
    if query in medical_queries_responses:
        raise HTTPException(status_code=400, detail="Query already exists.")
    
    # In-memory (idk if this actually works)
    medical_queries_responses[query] = response

    # Persist
    try:
        with open(DATA_FILE_PATH, "r") as file:
            content = file.read()
        
        new_content = content[:-2]
        new_content += f',\n    "{query}": "{response}"\n}}'

        with open(DATA_FILE_PATH, "w") as file:
            file.write(new_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to write to file.")

    return {"message": "Query-response pair added successfully."}

@router.get("/queries")
async def get_root():
    return {"Hello": "Queries"}