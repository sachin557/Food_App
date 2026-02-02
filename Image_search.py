import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not set")

llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

FOOD_DETECTION_PROMPT = """
You are a food recognition system.

Rules:
- Identify ONLY real foods visible in the image
- Estimate quantity ONLY using common visual portions
- Use STANDARD UNITS ONLY
- If unsure, choose the nearest standard serving
- DO NOT hallucinate foods

Allowed units:
- grams (g)
- milliliters (ml)
- cups
- pieces

Return ONLY comma-separated values in this format:
food_name (quantity unit)

Examples:
rice (1 cup), grilled chicken (150 g), boiled egg (1 piece)
milk (200 ml)

If no food detected, return empty string.
"""


def detect_foods_from_image(image_path: str) -> str:
    image = Image.open(image_path)

    response = llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[FOOD_DETECTION_PROMPT, image]
    )

    text = response.text.strip()

    # sanitize output
    text = text.replace("\n", " ").replace(".", "").strip()

    return text
