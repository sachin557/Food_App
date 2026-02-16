import os
import json
import re
from dotenv import load_dotenv
from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= ENV =================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found")

MAX_FOODS = 10

# ================= CLEAN TRANSCRIPT =================
def clean_transcript(text: str) -> str:
    junk = [
        "uh", "umm", "please", "can you",
        "i ate", "i had", "today", "yesterday"
    ]
    text = text.lower()
    for j in junk:
        text = text.replace(j, "")
    return text.strip()

# ================= JSON SAFETY =================
def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    raise HTTPException(status_code=500, detail="Invalid JSON from AI")

# ================= PROMPT =================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You are a nutrition extraction engine.

STRICT RULES:
- ALWAYS return at least ONE food if food is mentioned
- Ignore filler words
- Use standard serving if quantity unclear
- NEVER return empty foods list
- Max {MAX_FOODS} foods
- Return STRICT JSON ONLY

FORMAT:
{{
  "foods": [
    {{
      "food_name": "Egg",
      "quantity": "2 piece",
      "carbohydrates_g": number,
      "protein_g": number,
      "fat_g": number,
      "calories_kcal": number
    }}
  ]
}}
"""
    ),
    ("human", "Transcript: {food_input}")
])

parser = StrOutputParser()

# ================= CORE =================
def get_voice_nutrition(food_input: str) -> dict:
    food_input = clean_transcript(food_input)

    if not food_input:
        raise HTTPException(status_code=400, detail="Empty voice input")

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
        raise HTTPException(
            status_code=400,
            detail="No food detected from voice input"
        )

    total = {
        "carbohydrates_g": round(sum(f["carbohydrates_g"] for f in foods), 2),
        "protein_g": round(sum(f["protein_g"] for f in foods), 2),
        "fat_g": round(sum(f["fat_g"] for f in foods), 2),
        "calories_kcal": round(sum(f["calories_kcal"] for f in foods), 2),
    }

    return {
        "foods": foods,
        "total_nutrition": total
    }
