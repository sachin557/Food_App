from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from Type_Search import get_nutrition

app = FastAPI(title="Nutrition API")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ REQUEST MODEL ------------------
class FoodRequest(BaseModel):
    food_name: str

# ------------------ ROUTES ------------------
@app.post("/search-food")
async def search_food(data: FoodRequest):
    food_input = data.food_name.strip()

    if not food_input:
        raise HTTPException(status_code=400, detail="Food input cannot be empty")

    # ✅ run blocking LLM safely
    return await run_in_threadpool(get_nutrition, food_input)


@app.get("/")
def health():
    return {"status": "Nutrition API running"}
