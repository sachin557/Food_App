import os
import json
import re
from dotenv import load_dotenv
from google import genai
from PIL import Image
from fastapi import HTTPException

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not set")

llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

FOOD_DETECTION_PROMPT = """
You are a food recognition system.

Rules:
- Identify ONLY real foods visible in the image
- Estimate quantity using common visual portions
- Use ONLY these units: piece, gram, ml, cup
- If unsure, choose the closest standard portion
- DO NOT guess nutrition values
- DO NOT hallucinate foods

Return STRICT JSON ONLY in this format:

{
  "foods": [
    {
      "food_name": "string",
      "quantity_number": number,
      "quantity_unit": "piece | gram | ml | cup"
    }
  ]
}

If no food detected, return:
{ "foods": [] }
"""

def _extract_json(text: str) -> dict:
    """
    Safely extract JSON from Gemini output
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise HTTPException(
            status_code=500,
            detail="No JSON found in image model response"
        )

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid JSON returned by image model"
        )

def detect_foods_from_image(image_path: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")

        response = llm.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[FOOD_DETECTION_PROMPT, image]
        )

        raw_text = response.text.strip()
        return _extract_json(raw_text)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
