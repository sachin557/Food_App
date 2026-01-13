from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from Type_Search import get_nutrition
from Ai_coach_chat import gemini_fitness_chat

app = FastAPI(title="Nutrition & AI Fitness API")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ REQUEST MODELS ------------------
class FoodRequest(BaseModel):
    food_name: str


class ChatRequest(BaseModel):
    message: str


# ------------------ ROUTES ------------------

@app.post("/search-food")
async def search_food(data: FoodRequest):
    food_input = data.food_name.strip()

    if not food_input:
        raise HTTPException(status_code=400, detail="Food input cannot be empty")

    return await run_in_threadpool(get_nutrition, food_input)


@app.post("/ai-chat")
async def ai_chat(data: ChatRequest):
    user_message = data.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return await run_in_threadpool(gemini_fitness_chat, user_message)


@app.get("/")
def health():
    return {"status": "Nutrition & AI Fitness API running"}
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
