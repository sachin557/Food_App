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

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set")

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
        total["carbohydrates_g"] += food["carbohydrates_g"]
        total["protein_g"] += food["protein_g"]
        total["fat_g"] += food["fat_g"]
        total["calories_kcal"] += food["calories_kcal"]

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

# ===================== PROMPT =====================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are a professional nutrition assistant.

CRITICAL RULES:
- Fix spelling mistakes
- Use ONLY real foods
- DO NOT hallucinate foods
- DO NOT calculate totals
- DO NOT multiply values

Your job:
1. Extract food name
2. Extract quantity EXACTLY as user typed
3. Provide nutrition for ONE STANDARD UNIT ONLY

STANDARD UNIT EXAMPLES:
- Egg → 1 large egg
- Rice → 100 grams cooked
- Milk → 100 ml
- Chicken → 100 grams cooked

Return ONLY valid JSON in this EXACT format:

{{{{  
  "foods": [
    {{{{
      "food_name": "string",
      "quantity_number": number,
      "quantity_unit": "string",
      "standard_unit": "string",
      "nutrition_per_standard_unit": {{{{
        "carbohydrates_g": number,
        "protein_g": number,
        "fat_g": number,
        "calories_kcal": number
      }}}}
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
def invoke_with_retry(chain, payload, retries=2, delay=3):
    for attempt in range(retries):
        try:
            return chain.invoke(payload)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ===================== CALCULATION =====================
def apply_quantity_multiplier(food: dict) -> dict:
    qty = food["quantity_number"]
    base = food["nutrition_per_unit"]

    nutrition_unit = food.get("nutrition_unit", "").lower()
    quantity_unit = food.get("quantity_unit", "").lower()

    # 🔒 UNIT NORMALIZATION (minimal fix)
    factor = 1.0

    # If nutrition is per 100g or 100ml, normalize to per 1 unit
    if "100" in nutrition_unit:
        factor = qty / 100
    else:
        factor = qty

    return {
        "food_name": normalize_food_name(food["food_name"]),
        "quantity": f"{qty} {food['quantity_unit']}",
        "standard_unit": food.get("nutrition_unit"),
        "carbohydrates_g": round(base["carbohydrates_g"] * factor, 2),
        "protein_g": round(base["protein_g"] * factor, 2),
        "fat_g": round(base["fat_g"] * factor, 2),
        "calories_kcal": round(base["calories_kcal"] * factor, 2),
    }


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

    final_foods = [apply_quantity_multiplier(food) for food in foods]

    return {
        "result_type": "multiple" if len(final_foods) > 1 else "single",
        "serving_note": "Nutrition calculated using standard serving multiplied by user quantity",
        "foods": final_foods,
        "total_nutrition": calculate_total_nutrition(final_foods),
    }
