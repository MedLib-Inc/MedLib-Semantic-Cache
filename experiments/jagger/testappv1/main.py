# main.py
from fastapi import FastAPI
from testappv1.api import endpoints

# Initialize app
app = FastAPI(title="Jagger Test Cache v1")

app.include_router(endpoints.router)

# Simple health check
@app.get("/health")
def health_check():
    return {"status": "ok"}