# main.py
from fastapi import FastAPI
from testappv1.api.endpoints import router

app = FastAPI()

app.include_router(router)

# Basic router for testing
@app.get("/")
async def root():
    return {"message": "FastAPI is running"}