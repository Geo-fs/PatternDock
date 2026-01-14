from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from mn_event_ai.nlp.features import Vocab, build_vocab, make_text_field, pad_or_truncate, tokenize
from mn_event_ai.nlp.textcnn import TextCNN


LABELS_PATH = Path("data/labels/weak_labels.jsonl")
LABEL_MAP_PATH = Path("data/labels/label_map.json")

VOCAB_OUT = Path("data/labels/vocab.json")
MODEL_DIR = Path("models")
MODEL_OUT = MODEL_DIR / "textcnn.pt"
META_OUT = MODEL_DIR / "textcnn_meta.json"


@dataclass
class TrainConfig:
    seed: int = 42
    max_vocab: int = 20000
    min_freq: int = 2
    max_len: int = 240

    emb_dim: int = 128
    num_filters: int = 128
    kernel_sizes: Tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.2

    batch_size: int = 32
    epochs: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4

    val_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultiLabelTextDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Vocab, label_to_idx: Dict[str, int], max_len: int):
        self.rows = rows
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_len = max_len

        self.num_labels = len(label_to_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        text = r["text"]
        toks = tokenize(text)
        ids = self.vocab.encode(toks)
        ids = pad_or_truncate(ids, self.max_len, self.vocab.pad_id)
        x = torch.tensor(ids, dtype=torch.long)

        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for lab in r.get("labels", []):
            if lab in self.label_to_idx:
                y[self.label_to_idx[lab]] = 1.0

        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n = 0

    # For simple metrics: micro precision/recall/f1 at threshold 0.5
    tp = fp = fn = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        tp += int(((preds == 1) & (yb == 1)).sum().item())
        fp += int(((preds == 1) & (yb == 0)).sum().item())
        fn += int(((preds == 0) & (yb == 1)).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "loss": total_loss / max(n, 1),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def train() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    rows = load_jsonl(LABELS_PATH)
    label_to_idx = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))

    # Build vocab from all text (weak labels are noisy; vocab isn't)
    texts = [r["text"] for r in rows]
    vocab = build_vocab(texts, min_freq=cfg.min_freq, max_size=cfg.max_vocab)
    vocab.save(VOCAB_OUT)

    # Split train/val
    idxs = list(range(len(rows)))
    random.shuffle(idxs)
    cut = int(len(idxs) * (1.0 - cfg.val_split))
    train_rows = [rows[i] for i in idxs[:cut]]
    val_rows = [rows[i] for i in idxs[cut:]]

    train_ds = MultiLabelTextDataset(train_rows, vocab, label_to_idx, cfg.max_len)
    val_ds = MultiLabelTextDataset(val_rows, vocab, label_to_idx, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = TextCNN(
        vocab_size=len(vocab.itos),
        num_labels=len(label_to_idx),
        emb_dim=cfg.emb_dim,
        num_filters=cfg.num_filters,
        kernel_sizes=cfg.kernel_sizes,
        dropout=cfg.dropout,
        pad_id=vocab.pad_id,
    ).to(cfg.device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = running / max(n, 1)
        val_metrics = evaluate(model, val_loader, cfg.device)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_micro_f1": val_metrics["micro_f1"],
                }
            )
        )

        if val_metrics["micro_f1"] > best_f1:
            best_f1 = val_metrics["micro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Save best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), MODEL_OUT)

    meta = {
        "model": "TextCNN",
        "label_to_idx": label_to_idx,
        "vocab_path": str(VOCAB_OUT),
        "max_len": cfg.max_len,
        "config": {
            "emb_dim": cfg.emb_dim,
            "num_filters": cfg.num_filters,
            "kernel_sizes": list(cfg.kernel_sizes),
            "dropout": cfg.dropout,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "best_val_micro_f1": best_f1,
        "device": cfg.device,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved meta:  {META_OUT}")
    print(f"Saved vocab: {VOCAB_OUT}")


if __name__ == "__main__":
    train()
