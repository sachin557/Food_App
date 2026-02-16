import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Dict, List

from langchain_core.messages import HumanMessage
#from offline_data import offline_food_search
# -------- EXISTING IMPORTS --------
from Image_search import detect_foods_from_image
from speech_text import transcribe_audio
from voice_search import get_voice_nutrition
from Type_Search import get_nutrition
from Ai_coach_chat import ai_fitness_chat

# ✅ FIXED IMPORT (MATCH FILE NAME)
from height_langgraph import height_weight_app

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
    chat_history: Optional[List[Dict[str, str]]] = None

class HeightWeightRequest(BaseModel):
    height: float
    unit: str  # "cm" or "feet"
class OfflineFoodRequest(BaseModel):
    food_name: str

# ------------------ TEXT FOOD SEARCH ------------------
@app.post("/search-food")
async def search_food(data: FoodRequest):
    food_input = data.food_name.strip()
    if not food_input:
        raise HTTPException(status_code=400, detail="Food input cannot be empty")
    return await run_in_threadpool(get_nutrition, food_input)

# ------------------ VOICE → FOOD SEARCH ------------------
@app.post("/voice-food")
async def voice_food(file: UploadFile = File(...)):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = await run_in_threadpool(transcribe_audio, tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No speech detected")

        nutrition = await run_in_threadpool(get_voice_nutrition, text)

        return {
            "transcript": text,
            "foods": nutrition["foods"],
            "total_nutrition": nutrition["total_nutrition"],
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
)

# ------------------ AI FITNESS CHAT ------------------
@app.post("/ai-chat")
async def ai_chat(data: ChatRequest):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return await run_in_threadpool(
        ai_fitness_chat,
        data.message,
        data.food_context,
        data.chat_history or [],
    )

# ------------------ IMAGE → FOOD SEARCH ------------------
@app.post("/image-search")
async def image_search(file: UploadFile = File(...)):
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        detected = await run_in_threadpool(
            detect_foods_from_image,
            tmp_path
        )

        foods = detected.get("foods", [])
        if not foods:
            raise HTTPException(status_code=400, detail="No food detected")

        food_input = ", ".join(
            f'{f["quantity_number"]} {f["quantity_unit"]} {f["food_name"]}'
            for f in foods
        )

        nutrition = await run_in_threadpool(get_nutrition, food_input)

        return {
            "input_type": "image",
            "detected_foods": foods,
            **nutrition
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------ HEIGHT → WEIGHT ------------------
@app.post("/height-to-weight")
async def height_to_weight(data: HeightWeightRequest):
    height = data.height
    unit = data.unit.lower()

    if height <= 0:
        raise HTTPException(status_code=400, detail="Invalid height")

    # Convert feet → cm
    if unit == "feet":
        height = height * 30.48
    elif unit != "cm":
        raise HTTPException(status_code=400, detail="Invalid unit")

    try:
        result = height_weight_app.invoke({
            "messages": [HumanMessage(content=str(height))]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "height_cm": round(height, 2),
        "result": result["messages"][-1].content
    }

#@app.post("/offline-food-search")
#async def offline_food(data: OfflineFoodRequest):
 #   food = data.food_name.strip()

  #  if not food:
   #     raise HTTPException(status_code=400, detail="Food name required")

    #try:
     #   results = await run_in_threadpool(
      #      offline_food_search, food
       # )
    #except Exception as e:
     #   raise HTTPException(status_code=500, detail=str(e))

    #if not results:
     #   return {"foods": [], "message": "No similar food found"}

    #return {"foods": results}

# ------------------ HEALTH ------------------
@app.get("/health")
def health():
    return {"status": "ok"}
