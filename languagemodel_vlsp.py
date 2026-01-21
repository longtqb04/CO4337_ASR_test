import whisper

model = whisper.load_model("base")
result = model.transcribe("vlsp2020_train_set_02/database_sa1_Jan08_Mar19_cleaned_utt_0000000005-1.wav")
print(result["text"])