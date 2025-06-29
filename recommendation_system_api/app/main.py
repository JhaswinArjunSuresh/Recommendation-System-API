from fastapi import FastAPI
from pydantic import BaseModel
from .recommender import RecommenderSystem

app = FastAPI()
recommender = RecommenderSystem()

class UserQuery(BaseModel):
    user_id: str
    top_n: int = 5

class ItemQuery(BaseModel):
    item_id: str
    top_n: int = 5

@app.post("/recommend")
def recommend(query: UserQuery):
    recs = recommender.recommend_for_user(query.user_id, query.top_n)
    return {"user_id": query.user_id, "recommendations": recs}

@app.post("/similar")
def similar(query: ItemQuery):
    sims = recommender.similar_items(query.item_id, query.top_n)
    return {"item_id": query.item_id, "similar_items": sims}

@app.get("/health")
def health():
    return {"status": "ok"}

