from gtts import gTTS
import os
import uuid

AUDIO_DIR = "static/audio"

os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_audio(text: str):
    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(text=text, lang="en")
    tts.save(file_path)

    return f"/static/audio/{filename}"