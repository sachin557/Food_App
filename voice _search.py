import os
import json
import re
import time
from dotenv import load_dotenv
from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===================== ENV =====================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ===================== CONSTANTS =====================
MAX_FOODS = 10

# ===================== UTILS =====================
def normalize_food_name(name: str) -> str:
    return name.strip().title()


def calculate_total_nutrition(foods: list) -> dict:
    total = {
        "carbohydrates_g": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "calories_kcal": 0.0,
    }

    for food in foods:
        total["carbohydrates_g"] += float(food.get("carbohydrates_g", 0))
        total["protein_g"] += float(food.get("protein_g", 0))
        total["fat_g"] += float(food.get("fat_g", 0))
        total["calories_kcal"] += float(food.get("calories_kcal", 0))

    return {k: round(v, 2) for k, v in total.items()}


# ===================== SAFE JSON PARSER =====================
def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    raise HTTPException(status_code=500, detail="AI returned invalid JSON")


# ===================== LLM =====================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=700,
    timeout=40,
)

parser = StrOutputParser()

# ===================== PROMPT (CRITICAL FIX) =====================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are a professional nutrition assistant.

CRITICAL:
- Fix spelling mistakes
- Use ONLY real foods
- DO NOT hallucinate foods

Rules:
1. User may give multiple foods (MAX {MAX_FOODS}) and it might be coming from voice to text conversion
2. Use quantity if present, otherwise standard serving
3. Return nutrition per food
4. Return ONLY valid JSON

The response MUST be a JSON object with this EXACT structure:

{{{{ 
  "foods": [
    {{{{
      "food_name": "string",
      "quantity": "string",
      "carbohydrates_g": number,
      "protein_g": number,
      "fat_g": number,
      "calories_kcal": number
    }}}}
  ]
}}}}

DO NOT include any text outside JSON.
"""
        ),
        ("human", "Food input: {food_input}")
    ]
)

# ===================== RETRY HELPER =====================
def invoke_with_retry(chain, payload, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)


# ===================== CORE FUNCTION =====================
def get_nutrition(food_input: str) -> dict:
    # ---- FOOD COUNT LIMIT ----
    food_count = len([f for f in re.split(r",|and", food_input) if f.strip()])
    if food_count > MAX_FOODS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FOODS} foods allowed"
        )

    chain = prompt | llm | parser

    try:
        response = invoke_with_retry(
            chain,
            {"food_input": food_input},
            retries=3,
            delay=2,
        )
        data = safe_json_parse(response)
    except HTTPException:
        raise
    except Exception as e:
        print("❌ LLM ERROR:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Nutrition service temporarily unavailable"
        )

    foods = data.get("foods", [])
    if not foods:
        raise HTTPException(status_code=500, detail="No food detected")

    # Normalize names
    for food in foods:
        food["food_name"] = normalize_food_name(food.get("food_name", ""))

    return {
        "result_type": "multiple" if len(foods) > 1 else "single",
        "serving_note": "Nutrition calculated based on provided quantity or standard serving",
        "foods": foods,
        "total_nutrition": calculate_total_nutrition(foods),
    }
