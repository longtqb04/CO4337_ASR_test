import soundfile as sf
import librosa
import numpy as np
import unicodedata
import re

SAMPLE_RATE = 16000

def normalize_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )
    return audio.astype(np.float32)