import os
import json  # ✅ REQUIRED
from dotenv import load_dotenv
from google import genai
from PIL import Image
from fastapi import HTTPException  # ✅ REQUIRED

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

def detect_foods_from_image(image_path: str) -> dict:
    try:
        image = Image.open(image_path)

        response = llm.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[FOOD_DETECTION_PROMPT, image]
        )

        text = response.text.strip()

        return json.loads(text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Image model returned invalid JSON"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
