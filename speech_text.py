import os
from deepgram import DeepgramClient, PrerecordedOptions
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("❌ DEEPGRAM_API_KEY not set")

# ✅ Create client (NEW SDK)
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

def transcribe_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio:
            buffer_data = audio.read()

        # ✅ NEW API (this is the fix)
        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            smart_format=True,
        )

        response = dg_client.listen.prerecorded.transcribe_file(
            {"buffer": buffer_data},
            options,
        )

        # ✅ Extract transcript safely
        transcript = (
            response.results.channels[0]
            .alternatives[0]
            .transcript
        )

        return transcript.strip()

    except Exception as e:
        print("❌ SPEECH TRANSCRIPTION ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Speech transcription failed"
        )
