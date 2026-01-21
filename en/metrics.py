from jiwer import wer, cer, wil, mer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import torch

_embedder = SentenceTransformer("all-mpnet-base-v2")

def compute_basic_metrics(refs, hyps):
    return {
        "WER": wer(refs, hyps),
        "CER": cer(refs, hyps),
        "WIL": wil(refs, hyps),
        "MER": mer(refs, hyps),
    }

def compute_bert_score(refs, hyps):
    _, _, F1 = bert_score(hyps, refs, lang="en")
    return F1.mean().item()

def compute_semantic_error_rate(refs, hyps, threshold=0.85):
    emb_ref = _embedder.encode(refs, convert_to_tensor=True)
    emb_hyp = _embedder.encode(hyps, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb_ref, emb_hyp).diagonal()
    return (cos_sim < threshold).float().mean().item()