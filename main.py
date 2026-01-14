from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Dict, List

from Type_Search import get_nutrition
from Ai_coach_chat import gemini_fitness_chat  # Groq-based AI

# ------------------ APP INIT ------------------
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
    food_context: Optional[Dict] = None
    chat_history: Optional[List[Dict[str, str]]] = []


# ------------------ ROUTES ------------------

@app.post("/search-food")
async def search_food(data: FoodRequest):
    food_input = data.food_name.strip()

    if not food_input:
        raise HTTPException(
            status_code=400,
            detail="Food input cannot be empty"
        )

    return await run_in_threadpool(get_nutrition, food_input)


@app.post("/ai-chat")
async def ai_chat(data: ChatRequest):
    user_message = data.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # ✅ Forward message + memory to AI
    return await run_in_threadpool(
        gemini_fitness_chat,
        user_message,
        data.food_context,
        data.chat_history,
    )


# ------------------ HEALTH CHECK ------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
