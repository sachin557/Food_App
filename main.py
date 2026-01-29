import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Dict, List
from Image_search import detect_foods_from_image
from speech_text import transcribe_audio
from voice_search import get_voice_nutrition
from Type_Search import get_nutrition
from Ai_coach_chat import ai_fitness_chat

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
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print("📦 Audio size:", os.path.getsize(tmp_path))

        # 🎙 Speech → Text (Deepgram)
        text = await run_in_threadpool(transcribe_audio, tmp_path)
        print("🎙 Transcribed:", text)

        # ✅ If speech not understood, return safe empty response
        if not text.strip():
            return {
                "transcript": "",
                "foods": [],
                "total_nutrition": {}
            }

        # 🧠 Nutrition logic (Groq)
        nutrition = await run_in_threadpool(get_voice_nutrition, text)

        return {
            "transcript": text,
            "foods": nutrition["foods"],
            "total_nutrition": nutrition["total_nutrition"],
        }

    except Exception as e:
        print("❌ Voice food error:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Voice nutrition service unavailable"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ------------------ AI FITNESS CHAT ------------------
@app.post("/ai-chat")
async def ai_chat(data: ChatRequest):
    user_message = data.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return await run_in_threadpool(
        ai_fitness_chat,
        user_message,
        data.food_context,
        data.chat_history or [],
    )

@app.post("/image-search")
async def image_search(file: UploadFile = File(...)):
    tmp_path = None

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1️⃣ Detect food names from image
        food_names = await run_in_threadpool(
            detect_foods_from_image,
            tmp_path
        )

        if not food_names.strip():
            raise HTTPException(
                status_code=400,
                detail="No food detected in image"
            )

        # 2️⃣ Reuse existing nutrition pipeline
        nutrition_result = await run_in_threadpool(
            get_nutrition,
            food_names
        )

        return {
            "input_type": "image",
            "detected_foods": food_names,
            **nutrition_result
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ IMAGE SEARCH ERROR:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Image nutrition service unavailable"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------ HEALTH ------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
