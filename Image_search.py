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
- DO NOT estimate nutrition
- DO NOT guess quantities
- Return ONLY comma-separated food names
- If no food detected, return empty string

Example output:
rice, grilled chicken, boiled egg
"""

def detect_foods_from_image(image_path: str) -> str:
    image = Image.open(image_path)

    response = llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[FOOD_DETECTION_PROMPT, image]
    )

    text = response.text.strip()

    # sanitize output
    text = text.split("\n")[-1]
    text = text.replace(".", "").strip()

    return text
