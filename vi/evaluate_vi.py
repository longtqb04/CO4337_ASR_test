import argparse
import os
from datasets import load_dataset
from dataset_vi import load_audio, normalize_text
from whisper_model_vi import WhisperASR
from metrics_vi import (
    compute_basic_metrics,
    compute_bert_score,
    compute_semantic_error_rate
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model_name", default="base")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of test samples")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_dataset("json", data_files={"test": args.test})["test"]

    if args.samples is not None:
        ds = ds.select(range(min(args.samples, len(ds))))


    model = WhisperASR(args.model_name)

    refs, hyps = [], []

    for ex in ds:
        audio = load_audio(ex["audio"])
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