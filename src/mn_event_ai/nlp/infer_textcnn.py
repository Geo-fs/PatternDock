from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from mn_event_ai.nlp.features import Vocab, make_text_field, pad_or_truncate, tokenize
from mn_event_ai.nlp.textcnn import TextCNN


ARTICLES_IN = Path("data/processed/articles.jsonl")
VOCAB_PATH = Path("data/labels/vocab.json")
LABEL_MAP_PATH = Path("data/labels/label_map.json")
MODEL_PATH = Path("models/textcnn.pt")
META_PATH = Path("models/textcnn_meta.json")

PRED_OUT = Path("data/processed/article_predictions.jsonl")


@dataclass
class InferConfig:
    max_len: int = 240
    top_k: int = 3          # choose up to 2 labels per article by default
    min_prob: float = 0.35  # don't emit labels below this prob
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def invert_label_map(label_to_idx: Dict[str, int]) -> List[str]:
    # idx -> label
    inv = [None] * len(label_to_idx)
    for lab, i in label_to_idx.items():
        inv[i] = lab
    return inv


def build_model(vocab_size: int, num_labels: int, meta: Dict, pad_id: int) -> TextCNN:
    cfg = meta.get("config", {})
    kernel_sizes = tuple(cfg.get("kernel_sizes", [3, 4, 5]))
    model = TextCNN(
        vocab_size=vocab_size,
        num_labels=num_labels,
        emb_dim=int(cfg.get("emb_dim", 128)),
        num_filters=int(cfg.get("num_filters", 128)),
        kernel_sizes=kernel_sizes,
        dropout=float(cfg.get("dropout", 0.2)),
        pad_id=pad_id,
    )
    return model


@torch.no_grad()
def predict_one(model: TextCNN, vocab: Vocab, text: str, cfg: InferConfig) -> torch.Tensor:
    toks = tokenize(text)
    ids = vocab.encode(toks)
    ids = pad_or_truncate(ids, cfg.max_len, vocab.pad_id)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(cfg.device)  # [1, T]
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu()  # [L]
    return probs


def choose_labels(probs: torch.Tensor, idx_to_label: List[str], cfg: InferConfig) -> List[Tuple[str, float]]:
    vals, idxs = torch.topk(probs, k=min(cfg.top_k, probs.numel()))
    out: List[Tuple[str, float]] = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if v >= cfg.min_prob:
            out.append((idx_to_label[i], float(v)))

    # If nothing passes min_prob, classify as "other" (if available),
    # otherwise return empty.
    if not out:
        if "other" in idx_to_label:
            other_i = idx_to_label.index("other")
            return [("other", float(probs[other_i].item()))]
        return []

    return out

def main() -> None:
    cfg = InferConfig()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model: {MODEL_PATH}. Train first.")
    if not VOCAB_PATH.exists():
        raise SystemExit(f"Missing vocab: {VOCAB_PATH}. Train first.")
    if not LABEL_MAP_PATH.exists():
        raise SystemExit(f"Missing label map: {LABEL_MAP_PATH}. Run weak_label first.")
    if not ARTICLES_IN.exists():
        raise SystemExit(f"Missing articles: {ARTICLES_IN}. Run pipeline first.")

    vocab = Vocab.load(VOCAB_PATH)
    label_to_idx = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    idx_to_label = invert_label_map(label_to_idx)

    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {"config": {}}
    cfg.max_len = int(meta.get("max_len", cfg.max_len))

    model = build_model(len(vocab.itos), len(label_to_idx), meta, vocab.pad_id)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    rows = load_jsonl(ARTICLES_IN)

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PRED_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            text = make_text_field(r.get("title", ""), r.get("summary", ""))
            probs = predict_one(model, vocab, text, cfg)
            chosen = choose_labels(probs, idx_to_label, cfg)

            out = {
                "id": r["id"],
                "source_id": r.get("source_id"),
                "published_at": r.get("published_at"),
                "title": r.get("title"),
                "canonical_url": r.get("canonical_url"),
                "pred_labels": [lab for lab, _p in chosen],
                "pred_probs": {lab: p for lab, p in chosen},
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote": str(PRED_OUT), "articles": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
