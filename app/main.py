from fastapi import FastAPI
from .routers import queries, database

app = FastAPI()
app.include_router(queries.router)
app.include_router(database.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}