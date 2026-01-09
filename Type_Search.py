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

NORMALIZE_PROMPT ="""
You are a professional nutrition assistant.

CRITICAL INSTRUCTIONS (DO NOT IGNORE):
1. Correct spelling mistakes in food names.
2. Identify the MOST LIKELY intended real food.
3. Use STANDARD, WELL-KNOWN food names only.
4. DO NOT invent food names.
5. DO NOT split words incorrectly.
6. If input is unclear, infer the closest common food.
7. Normalize food names (e.g., "msala fosa" → "Masala Dosa").

Rules:
1. User may give ONE or MULTIPLE foods.
2. Handle quantities if present.
3. If quantity is missing, assume standard serving.
4. Calculate nutrition per food.
5. Calculate TOTAL nutrition.
6. Return ONLY valid JSON.

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
