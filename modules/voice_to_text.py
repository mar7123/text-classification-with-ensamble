import pandas as pd
from openai import OpenAI

from config import Config

api_key = Config.OPEN_AI_KEY
client = OpenAI(api_key=api_key)
# Supported file types
supported_file_types = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}

# Max file size in bytes (25 MB)
max_file_size = 25 * 1024 * 1024


def transcribe_audio(file_path: str):
    audio_file = open(file_path, "rb")
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="id"
    )
    return transcription


def save_transcription(transcription, file_path):
    df = pd.DataFrame({"response": [transcription.text]})
    print(df)
    df.to_excel(file_path, index=False)

    print(f"Transcription saved to {file_path}")
