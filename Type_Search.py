from dotenv import load_dotenv
import os
import json
import re
from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===================== ENV =====================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("Groq_api")

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

# ===================== LLM =====================

llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

PROMPT_TEMPLATE = """
You are a professional nutrition assistant.

CRITICAL:
- Fix spelling mistakes (e.g. "msala dosa" → "Masala Dosa")
- Use ONLY real, common food names
- DO NOT split words incorrectly

Rules:
1. User may give ONE or MULTIPLE foods.
2. Use quantity if present, otherwise standard serving.
3. Return nutrition per food.
4. Return ONLY valid JSON.

JSON FORMAT:
{
  "foods": [
    {
      "food_name": string,
      "quantity": string,
      "carbohydrates_g": number,
      "protein_g": number,
      "fat_g": number,
      "calories_kcal": number
    }
  ]
}

Food input:
{food_input}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("user", "{food_input}")
    ]
)

# ===================== CORE FUNCTION =====================

def get_nutrition(food_input: str) -> dict:
    chain = prompt | llm | parser
    response = chain.invoke({"food_input": food_input})

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        raise HTTPException(500, "Invalid AI response")

    foods = data.get("foods", [])
    if not foods:
        raise HTTPException(500, "No food detected")

    for food in foods:
        food["food_name"] = normalize_food_name(food["food_name"])
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
