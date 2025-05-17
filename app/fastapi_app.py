from fastapi import FastAPI, Query
from app.pipeline import answer

app = FastAPI()
@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    return answer(q, k)