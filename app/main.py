import logging
from fastapi import FastAPI
from .routers import queries, database


# Set up logging configuration
# Run with 'uvicorn app.main:app --reload --log-level info' to ensure logging enabled
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format
)

app = FastAPI()
app.include_router(queries.router)
app.include_router(database.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}