import os
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DG_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set")

client = DeepgramClient(DG_API_KEY)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio using Deepgram SDK v5.x
    """
    with open(file_path, "rb") as audio:
        response = client.listen.rest.transcribe_file(
            audio,
            {
                "model": "nova-2",
                "language": "en",
                "punctuate": True,
                "smart_format": True,
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
            }
        )

    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    print("🧠 Deepgram transcript:", repr(transcript))

    return transcript.strip() if transcript else ""
