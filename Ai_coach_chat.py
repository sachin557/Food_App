import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a professional AI Fitness Coach and Nutrition Assistant.

Rules:
- Give workout + diet advice
- Use bullet points
- Be motivating
- Avoid medical claims
"""

def ai_fitness_chat(user_message, food_context=None, chat_history=None):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # ✅ Inject food memory ONCE
        if food_context:
            messages.append({
                "role": "system",
                "content": f"""
User food history (use this to personalize advice):
{food_context}
"""
            })

        # ✅ FIX ROLE NAMES + PRESERVE ORDER
        if chat_history:
            for msg in chat_history:
                role = msg["role"]
                if role == "ai":
                    role = "assistant"   # 🔥 CRITICAL FIX

                messages.append({
                    "role": role,
                    "content": msg["text"]
                })
        else:
            # Only add user message if no history exists
            messages.append({
                "role": "user",
                "content": user_message
            })

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=700,
        )

        return {
            "reply": completion.choices[0].message.content.strip()
        }

    except Exception as e:
        print("Groq error:", e)
        return {
            "reply": "⚠️ AI is temporarily unavailable"
        }
