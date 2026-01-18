import os
import json
from fastapi import HTTPException
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= ENV =================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found")

MAX_FOODS = 10

# ================= JSON SAFETY =================
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
    raise HTTPException(status_code=500, detail="Invalid JSON from AI")

# ================= PROMPT =================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You are a professional nutrition assistant.

Rules:
1. Max {MAX_FOODS} foods
2. Fix spelling mistakes
3. Use real foods only
4. Use quantity if present else standard serving
5. Return ONLY JSON

Format:
{{
  "foods": [
    {{
      "food_name": "",
      "quantity": "",
      "carbohydrates_g": 0,
      "protein_g": 0,
      "fat_g": 0,
      "calories_kcal": 0
    }}
  ]
}}
"""
    ),
    ("human", "Food input: {food_input}")
])

parser = StrOutputParser()

# ================= CORE FUNCTION =================
def get_voice_nutrition(food_input: str) -> dict:
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=700,
            timeout=40,
        )

        chain = prompt | llm | parser
        response = chain.invoke({"food_input": food_input})

        data = safe_json_parse(response)
        foods = data.get("foods", [])

        if not foods:
            return {
                "foods": [],
                "total_nutrition": {}
            }

        total = {
            "carbohydrates_g": round(sum(f.get("carbohydrates_g", 0) for f in foods), 2),
            "protein_g": round(sum(f.get("protein_g", 0) for f in foods), 2),
            "fat_g": round(sum(f.get("fat_g", 0) for f in foods), 2),
            "calories_kcal": round(sum(f.get("calories_kcal", 0) for f in foods), 2),
        }

        return {
            "foods": foods,
            "total_nutrition": total
        }

    except Exception as e:
        print("❌ GROQ ERROR:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Nutrition service temporarily unavailable"
        )
