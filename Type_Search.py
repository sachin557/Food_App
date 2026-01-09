from dotenv import load_dotenv
import os
import json
import re

from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("Groq_api")

def extract_quantity(text: str):
    """
    Extracts quantity like:
    100g, 1 cup, 2 eggs, 250 ml
    """
    match = re.search(
        r"(\d+(\.\d+)?\s?(g|gm|grams|kg|ml|cup|cups|tbsp|tsp|pieces|piece|eggs))",
        text.lower()
    )
    if match:
        return match.group(1)
    return "Standard serving"

def has_quantity(text: str) -> bool:
    return extract_quantity(text) != "Standard serving"

llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

PROMPT_TEMPLATE = """
You are a professional nutrition assistant.

Rules:
1. User may give ONE or MULTIPLE foods in one input.
2. Split multiple foods correctly.
3. If quantity is mentioned for a food, use it.
4. If quantity is NOT mentioned, assume standard serving size.
5. Calculate nutrition for EACH food separately.
6. Calculate TOTAL nutrition by summing all foods.
7. Return ONLY valid JSON. No explanation. No markdown.

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("user", "{food_input}")
    ]
)

def get_nutrition(food_input: str) -> dict:
    """
    Core nutrition logic.
    Can be reused by FastAPI, CLI, batch jobs, tests, etc.
    """

    chain = prompt | llm | parser
    response = chain.invoke({"food_input": food_input})

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
        "food_input": food_input.title(),
        "serving_note": (
            "Based on user provided quantity"
            if has_quantity(food_input)
            else "Based on standard serving size"
        ),
        "foods": data["foods"],
        "total_nutrition": data["total_nutrition"]
    }
