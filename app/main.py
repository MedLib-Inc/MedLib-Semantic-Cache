from fastapi import FastAPI
from .routers import queries

app = FastAPI()
app.include_router(queries.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}