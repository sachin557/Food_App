import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

FOOD_DETECTION_PROMPT = """
You are a food recognition system.

Rules:
- Identify ONLY real foods visible in the image
- DO NOT estimate nutrition
- DO NOT guess quantities
- Return ONLY comma-separated food names
- If multiple foods exist, list all

Example output:
"rice, grilled chicken, boiled egg"

Return ONLY the food names, nothing else.
"""

def detect_foods_from_image(image_path: str) -> str:
    image = Image.open(image_path)

    response = llm.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[FOOD_DETECTION_PROMPT, image]
    )

    return response.text.strip()
