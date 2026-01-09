from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Type_Search import get_nutrition


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodRequest(BaseModel):
    food_name: str



@app.post("/search-food")
def search_food(data: FoodRequest):
    if not data.food_name.strip():
        return {"error": "Input cannot be empty"}

    return get_nutrition(data.food_name)

@app.get("/")
def health():
    return {"status": "Nutrition API running"}
