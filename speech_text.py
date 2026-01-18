import os
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DG_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set")

deepgram = DeepgramClient(api_key=DG_API_KEY)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio using Deepgram API (EC2-safe)
    """
    with open(file_path, "rb") as audio:
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            audio,
            {
                "punctuate": True,
                "language": "en",
                "model": "nova-2"
            }
        )

    return response["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
