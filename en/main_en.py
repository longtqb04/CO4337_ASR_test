import argparse
import os
from datasets import load_dataset
from dataset import load_audio, normalize_text
from whisper_model import WhisperASR
from metrics import (
    compute_basic_metrics,
    compute_bert_score,
    compute_semantic_error_rate
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model_name", default="base.en")
    return parser.parse_args()

def main():
    args = parse_args()

    # Change to workspace root directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    ds = load_dataset("json", data_files={"test": args.test})["test"]

    model = WhisperASR(args.model_name)

    refs, hyps = [], []

    for ex in ds:
        audio = load_audio(ex)
        hyp = model.transcribe(audio)
        ref = ex["text"]

        refs.append(normalize_text(ref))
        hyps.append(normalize_text(hyp))

    print("===== ASR Evaluation (Whisper) =====")

    basic = compute_basic_metrics(refs, hyps)
    for k, v in basic.items():
        print(f"{k}: {v:.4f}")

    print(f"BERTScore-F1: {compute_bert_score(refs, hyps):.4f}")
    print(f"Semantic Error Rate: {compute_semantic_error_rate(refs, hyps):.4f}")

if __name__ == "__main__":
    main()