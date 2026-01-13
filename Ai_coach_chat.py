import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ------------------ GEMINI CONFIG ------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
You are a professional AI Fitness Coach and Nutrition Assistant.

Your role:
- Create personalized fitness plans
- Suggest workouts (gym, home, cardio, strength, yoga)
- Give diet & calorie guidance
- Be clear, motivating, and practical
- Use bullet points
- Avoid medical claims
- Ask follow-up questions if needed

Tone:
- Friendly
- Motivational
- Simple language
"""
)

# ------------------ AI CHAT FUNCTION ------------------
def gemini_fitness_chat(user_message: str) -> dict:
    try:
        response = model.generate_content(
            f"""
User request:
{user_message}

Respond as a fitness coach with a clear plan.
"""
        )

        return {
            "reply": response.text.strip()
        }

    except Exception as e:
        return {
            "reply": "⚠️ Sorry, I couldn't generate a fitness plan right now.",
            "error": str(e)
        }
