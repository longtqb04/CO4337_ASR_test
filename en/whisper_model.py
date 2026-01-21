import whisper

class WhisperASR:
    def __init__(self, model_name="base.en", device=None):
        self.model = whisper.load_model(model_name)
        if device:
            self.model.to(device)

    def transcribe(self, audio):
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False
        )
        return result["text"]
    
# Example usage:
# asr = WhisperASR(model_name="small.en", device="cuda")
# transcription = asr.transcribe("path_to_audio_file.wav")