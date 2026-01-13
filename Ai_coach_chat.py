import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ------------------ GROQ CLIENT ------------------
client = Groq(
    api_key=os.getenv("Groq_api")
)

SYSTEM_PROMPT = """
You are a professional AI Fitness Coach and Nutrition Assistant.

Your responsibilities:
- Create personalized fitness plans
- Suggest workouts (gym, home, cardio, strength, yoga)
- Provide diet & calorie guidance
- Be motivating, clear, and practical
- Use bullet points
- Avoid medical advice
- Ask follow-up questions if needed

Tone:
- Friendly
- Motivational
- Simple language
"""

# ------------------ AI CHAT FUNCTION ------------------
def gemini_fitness_chat(user_message: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.7,
            max_tokens=600,
        )

        reply_text = completion.choices[0].message.content

        return {"reply": reply_text.strip()}

    except Exception as e:
        print("Groq error:", e)
        return {
            "reply": "⚠️ Sorry, I couldn't generate a fitness plan right now."
        }
