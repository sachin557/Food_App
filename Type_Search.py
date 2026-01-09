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
        r"(\d+(\.\d+)?\s?(g|gm|grams|kg|ml|cup|cups|tbsp|tsp|pieces|piece|eggs))",
        text.lower()
    )
    return match.group(1) if match else "Standard serving"

def has_quantity(text: str) -> bool:
    return extract_quantity(text) != "Standard serving"

def normalize_food_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split())

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

# ===================== PROMPT =====================

PROMPT_TEMPLATE = """
You are a professional nutrition assistant.

CRITICAL RULES (DO NOT BREAK):
1. ALWAYS split foods into separate items.
2. NEVER merge multiple foods into one.
3. If input contains "and", commas, or multiple foods → return MULTIPLE food objects.
4. Correct spelling mistakes (e.g., "msala dosa" → "Masala Dosa").
5. Use standard food names only.

Examples:
Input: "2 eggs and masala dosa"
Output foods:
- Eggs
- Masala Dosa

Input: "neer dosa masala dosa"
Output foods:
- neer dosa
- Masala Dosa

Input: "apple, banana"
Output foods:
- Apple
- Banana

Rules:
- Use quantity if provided, else assume standard serving.
- Calculate nutrition PER FOOD.
- Return ONLY valid JSON.

STRICT JSON FORMAT:

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
        raise HTTPException(status_code=500, detail="Invalid response from AI model")

    foods = data.get("foods", [])
    if not foods:
        raise HTTPException(status_code=500, detail="No food detected")

    # Normalize foods
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
