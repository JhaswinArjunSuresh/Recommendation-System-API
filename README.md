# 🎯 Recommendation System API

Matrix factorization (ALS) on implicit feedback data, served via FastAPI.

## Endpoints
- `POST /recommend` with `{ "user_id": "u1", "top_n": 5 }`
- `POST /similar` with `{ "item_id": "i1", "top_n": 5 }`

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

