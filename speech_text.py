import whisper

# Load model ONCE (important)
# Options:
# tiny  -> fastest, least accurate
# base  -> best balance (recommended)
# small -> better accuracy, slower

model = whisper.load_model("base")

def transcribe_audio(file_path: str) -> str:
    """
    Converts speech audio to text using Whisper
    """
    result = model.transcribe(
        file_path,
        language="en",
        fp16=False,  # required on CPU
    )
    return result["text"].strip()
