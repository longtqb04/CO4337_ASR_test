import argparse
import os
import torch
from torch.utils.data import DataLoader
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer as jiwer_wer
from jiwer import cer as jiwer_cer

SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model_name", default="patrickvonplaten/wav2vec2_tiny_random")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--debug_small", action="store_true")
    return parser.parse_args()


def is_valid_sample(example):
    input_len = len(example["input_values"])
    label_ids = example["labels"]
    label_len = sum(l != -100 for l in label_ids)

    if label_len == 0:
        return False

    return input_len > label_len



def normalize_text(text):
    import re
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_audio(example):
    """Load & resample audio"""
    audio_info = example["audio"]

    if isinstance(audio_info, dict):
        path = audio_info["path"]
    else:
        path = audio_info

    audio, sr = sf.read(path)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    example["audio"] = audio
    return example


def prepare_dataset(batch, processor):
    audio_arrays = batch["audio"]

    # inputs (audio → features)
    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=False
    )

    batch["input_values"] = inputs.input_values

    # labels (text → token ids)
    texts = [normalize_text(t) for t in batch["text"]]

    with processor.as_target_processor():
        labels = processor.tokenizer(
            texts,
            padding=True
        ).input_ids

    batch["labels"] = labels
    return batch


def collate_fn(batch):
    input_values = [torch.tensor(b["input_values"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

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


if __name__ == "__main__":

    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        pad_token_id = processor.tokenizer.pad_token_id,
        vocab_size = len(processor.tokenizer)
    )
    model.config.ctc_zero_infinity = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_files = {
        "train": args.train,
        "dev": args.dev,
        "test": args.test
    }

    ds = load_dataset("json", data_files=data_files)

    # small debugging mode
    if args.debug_small:
        for split in ds.keys():
            ds[split] = ds[split].select(range(min(50, len(ds[split]))))

    # audio loading
    for split in ds.keys():
        ds[split] = ds[split].map(load_audio)

    # feature conversion
    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda b: prepare_dataset(b, processor),
            batched=True,
            batch_size=4,
            remove_columns=["audio", "text"]
        )

    for split in ds.keys():
        ds[split] = ds[split].filter(is_valid_sample)

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds["dev"], batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")

        # validation
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


    # test run
    model.eval()
    wer_scores = []
    cer_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            wer_scores.append(compute_wer_batch(logits, batch["labels"], processor))
            cer_scores.append(compute_cer_batch(logits, batch["labels"], processor))

    print(f"Final Test WER: {np.mean(wer_scores):.4f}")
    print(f"Final Test CER: {np.mean(cer_scores):.4f}")

    os.makedirs("output_model", exist_ok=True)
    model.save_pretrained("output_model")
    processor.save_pretrained("output_model")

    print("Model saved → output_model/")