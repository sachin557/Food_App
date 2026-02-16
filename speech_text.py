import os
from dotenv import load_dotenv
from fastapi import HTTPException
from deepgram import Deepgram

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY not set")

dg = Deepgram(DEEPGRAM_API_KEY)

def transcribe_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}

            response = dg.transcription.prerecorded(
                source,
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
