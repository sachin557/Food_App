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

def extract_quantity(text: str):
    match = re.search(
        r"(\d+(\.\d+)?\s?(g|gm|grams|kg|ml|cup|cups|tbsp|tsp|pieces|piece|eggs))",
        text.lower()
    )
    if match:
        return match.group(1)
    return "Standard serving"

def has_quantity(text: str) -> bool:
    return extract_quantity(text) != "Standard serving"

# ===================== LLM =====================

llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

# ===================== NORMALIZATION PROMPT =====================

NORMALIZE_PROMPT = """
You are a food name normalization expert.

Rules:
1. Correct spelling mistakes.
2. If words together form ONE known dish, treat as ONE food.
   Examples:
   - "masala dosa" → ONE food
   - "chicken biryani" → ONE food
3. Split foods ONLY if they are clearly separate using:
   "and", ",", "+"
4. Return ONLY valid JSON.

FORMAT:
{
  "is_single_food": boolean,
  "foods": [string]
}

User input:
{food_input}
"""

normalize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", NORMALIZE_PROMPT),
        ("user", "{food_input}")
    ]
)

# ===================== NUTRITION PROMPT =====================

NUTRITION_PROMPT = """
You are a professional nutrition assistant.

Rules:
1. Food names are already normalized.
2. Use quantity if provided, otherwise assume standard serving size.
3. Calculate nutrition for EACH food separately.
4. Calculate TOTAL nutrition by summing all foods.
5. Return ONLY valid JSON.

STRICT JSON FORMAT:

{{
  "foods": [
    {{
      "food_name": string,
      "quantity": string,
      "carbohydrates_g": number,
      "protein_g": number,
      "fat_g": number,
      "calories_kcal": number
    }}
  ],
  "total_nutrition": {{
    "carbohydrates_g": number,
    "protein_g": number,
    "fat_g": number,
    "calories_kcal": number
  }}
}}

Food input:
{food_input}
"""

nutrition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", NUTRITION_PROMPT),
        ("user", "{food_input}")
    ]
)

# ===================== NORMALIZE INPUT =====================

def normalize_food_input(food_input: str) -> dict:
    chain = normalize_prompt | llm | parser
    response = chain.invoke({"food_input": food_input})

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Safe fallback
        return {
            "is_single_food": True,
            "foods": [food_input]
        }

# ===================== CORE FUNCTION =====================

def get_nutrition(food_input: str) -> dict:
    normalized = normalize_food_input(food_input)

    foods_for_llm = " and ".join(normalized["foods"])

    chain = nutrition_prompt | llm | parser
    response = chain.invoke({"food_input": foods_for_llm})

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid response from AI model"
        )

    if "foods" not in data or "total_nutrition" not in data:
        raise HTTPException(
            status_code=500,
            detail="Malformed nutrition response"
        )

    for food in data["foods"]:
        if not food.get("quantity"):
            food["quantity"] = extract_quantity(food["food_name"])

    return {
        "result_type": "single" if normalized["is_single_food"] else "multiple",
        "food_input": food_input.title(),
        "serving_note": (
            "Based on user provided quantity"
            if has_quantity(food_input)
            else "Based on standard serving size"
        ),
        "foods": data["foods"],
        "total_nutrition": data["total_nutrition"]
    }
