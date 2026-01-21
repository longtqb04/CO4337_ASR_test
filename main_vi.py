import argparse
import os
import torch
from torch.utils.data import DataLoader
import soundfile as sf
import librosa
import numpy as np
import unicodedata
import re
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer as jiwer_wer
from jiwer import cer as jiwer_cer

SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to vlsp_train.jsonl")
    parser.add_argument("--model_name", default="patrickvonplaten/wav2vec2_tiny_random")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--debug_small", action="store_true")
    return parser.parse_args()


def normalize_text_ctc_vi(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_audio(example):
    audio_info = example["audio"]
    path = audio_info["path"] if isinstance(audio_info, dict) else audio_info

    audio, sr = sf.read(path)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    example["audio"] = audio
    return example


def prepare_dataset(batch, processor):
    inputs = processor(
        batch["audio"],
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=False
    )

    batch["input_values"] = inputs.input_values

    texts = [normalize_text_ctc_vi(t) for t in batch["text"]]

    with processor.as_target_processor():
        labels = processor.tokenizer(
            texts,
            padding=True
        ).input_ids

    batch["labels"] = labels
    return batch


def is_valid_sample(example):
    input_len = len(example["input_values"])
    label_len = sum(l != -100 for l in example["labels"])
    return label_len > 0 and input_len > label_len


def collate_fn(batch):
    input_values = [torch.tensor(b["input_values"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    input_values = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_values": input_values,
        "labels": labels
    }


def compute_wer_batch(logits, labels, processor):
    pred_ids = np.argmax(logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    labels_cpu = labels.cpu().numpy()
    labels_cpu[labels_cpu == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_cpu, group_tokens=False)

    return jiwer_wer(label_str, pred_str)


def compute_cer_batch(logits, labels, processor):
    pred_ids = np.argmax(logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    labels_cpu = labels.cpu().numpy()
    labels_cpu[labels_cpu == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_cpu, group_tokens=False)

    return jiwer_cer(label_str, pred_str)

def evaluate(loader, model, processor, device):
    model.eval()

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits

            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = processor.batch_decode(pred_ids)

            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            ref_str = processor.batch_decode(labels, group_tokens=False)

            all_preds.extend(pred_str)
            all_refs.extend(ref_str)

    wer = jiwer_wer(all_refs, all_preds)
    cer = jiwer_cer(all_refs, all_preds)

    return wer, cer

if __name__ == "__main__":

    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    model.config.ctc_zero_infinity = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load single VLSP file
    ds = load_dataset("json", data_files={"train": args.train})
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)

    if args.debug_small:
        ds["train"] = ds["train"].select(range(min(50, len(ds["train"]))))
        ds["test"] = ds["test"].select(range(min(20, len(ds["test"]))))

    # Load audio
    for split in ds.keys():
        ds[split] = ds[split].map(load_audio)

    # Feature extraction
    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda b: prepare_dataset(b, processor),
            batched=True,
            batch_size=4,
            remove_columns=["audio", "text"]
        )

    # Filter bad samples
    for split in ds.keys():
        ds[split] = ds[split].filter(is_valid_sample)

    # Dataloaders
    train_loader = DataLoader(
        ds["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        ds["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        wer_scores = []
        cer_scores = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits.cpu().numpy()

                wer_scores.append(compute_wer_batch(logits, batch["labels"], processor))
                cer_scores.append(compute_cer_batch(logits, batch["labels"], processor))

        print(f"[Epoch {epoch+1}] WER: {np.mean(wer_scores):.4f}")
        print(f"[Epoch {epoch+1}] CER: {np.mean(cer_scores):.4f}")

    final_wer, final_cer = evaluate(
        val_loader,
        model,
        processor,
        device
    )

    print(f"Final Validation WER: {final_wer:.4f}")
    print(f"Final Validation CER: {final_cer:.4f}")

    os.makedirs("output_vlsp", exist_ok=True)
    model.save_pretrained("output_vlsp")
    processor.save_pretrained("output_vlsp")

    print("Model saved → output_vlsp/")