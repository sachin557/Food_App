import os
from dotenv import load_dotenv
from fastapi import HTTPException
from deepgram import DeepgramClient

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY not set")

# ✅ Correct initialization
dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

def transcribe_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio:
            audio_bytes = audio.read()

        # ✅ THIS WORKS WITH YOUR INSTALLED SDK
        response = dg_client.listen.transcribe_file(
            {"buffer": audio_bytes},
            {
                "model": "nova-2",
                "language": "en",
                "smart_format": True,
            },
        )

        transcript = (
            response["results"]["channels"][0]
            ["alternatives"][0]["transcript"]
        )

        return transcript.strip()

    except Exception as e:
        print("❌ DEEPGRAM ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Speech transcription failed"
        )
