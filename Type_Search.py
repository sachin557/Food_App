import os
import json
import re
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
def extract_quantity(text: str) -> str:
    match = re.search(
        r"(\d+(\.\d+)?\s?(g|gm|grams|kg|ml|cup|cups|tbsp|tsp|pieces|piece|eggs|slice))",
        text.lower()
    )
    return match.group(1) if match else "Standard serving"


def has_quantity(text: str) -> bool:
    return extract_quantity(text) != "Standard serving"


def normalize_food_name(name: str) -> str:
    return name.strip().title()


def calculate_total_nutrition(foods: list) -> dict:
    total = {
        "carbohydrates_g": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "calories_kcal": 0.0
    }

    for food in foods:
        total["carbohydrates_g"] += float(food.get("carbohydrates_g", 0))
        total["protein_g"] += float(food.get("protein_g", 0))
        total["fat_g"] += float(food.get("fat_g", 0))
        total["calories_kcal"] += float(food.get("calories_kcal", 0))

    return {k: round(v, 2) for k, v in total.items()}


# ---------- SAFE JSON PARSER ----------
def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt recovery from partial output
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
    max_tokens=800,
    timeout=60,
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are a professional nutrition assistant.

CRITICAL:
- Fix spelling mistakes (e.g. "msala dosa" → "Masala Dosa")
- Use ONLY real, common food names
- DO NOT hallucinate foods

Rules:
1. User may give multiple foods (MAX {MAX_FOODS}).
2. If more than {MAX_FOODS} foods are detected, return ONLY the first {MAX_FOODS}.
3. Use quantity if present, otherwise standard serving.
4. Return nutrition per food.
5. Return ONLY valid JSON.

JSON FORMAT (STRICT):

{{
  "foods": [
    {{
      "food_name": "string",
      "quantity": "string",
      "carbohydrates_g": number,
      "protein_g": number,
      "fat_g": number,
      "calories_kcal": number
    }}
  ]
}}
"""
        ),
        ("human", "Food input: {food_input}")
    ]
)

# ===================== CORE FUNCTION =====================
def get_nutrition(food_input: str) -> dict:
    # ---------- HARD FOOD LIMIT ----------
    food_count = len([f for f in re.split(r",|and", food_input) if f.strip()])
    if food_count > MAX_FOODS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FOODS} foods allowed per request"
        )

    chain = prompt | llm | parser

    try:
        response = chain.invoke({"food_input": food_input})
        data = safe_json_parse(response)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Nutrition service temporarily unavailable"
        )

    foods = data.get("foods", [])
    if not foods:
        raise HTTPException(status_code=500, detail="No food detected")

    for food in foods:
        food["food_name"] = normalize_food_name(food.get("food_name", ""))
        if not food.get("quantity"):
            food["quantity"] = extract_quantity(food["food_name"])

    total_nutrition = calculate_total_nutrition(foods)

    return {
        "result_type": "multiple" if len(foods) > 1 else "single",
        "serving_note": (
            "Based on user provided quantity"
            if has_quantity(food_input)
            else "Based on standard serving size"
        ),
        "foods": foods,
        "total_nutrition": total_nutrition
    }
