import soundfile as sf
import librosa
import numpy as np

SAMPLE_RATE = 16000

def normalize_text(text):
    import re
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_audio(example):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, sr, SAMPLE_RATE)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32)