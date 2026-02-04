import os
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

DG_API_KEY = os.getenv("DG_API_KEY")

client = DeepgramClient(api_key=DG_API_KEY)

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        response = client.listen.prerecorded.v("1").transcribe_file(
            f,
            {
                "model": "nova-2",
                "language": "en",
                "smart_format": True,
            },
        )

    return response["results"]["channels"][0]["alternatives"][0]["transcript"]
