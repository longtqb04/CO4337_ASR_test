import whisper

model = whisper.load_model("base.en")
result = model.transcribe("LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
print(result["text"])