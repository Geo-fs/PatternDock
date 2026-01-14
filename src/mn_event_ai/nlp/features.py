from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos, "pad_token": self.pad_token, "unk_token": self.unk_token}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Vocab":
        payload = json.loads(path.read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(stoi=stoi, itos=itos, pad_token=payload["pad_token"], unk_token=payload["unk_token"])


def build_vocab(
    texts: Iterable[str],
    min_freq: int = 2,
    max_size: int = 50000,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> Vocab:
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))

    # Reserve 0/1 for pad/unk
    itos = [pad_token, unk_token]
    for tok, freq in counts.most_common():
        if freq < min_freq:
            break
        if tok in (pad_token, unk_token):
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_token=pad_token, unk_token=unk_token)


def pad_or_truncate(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def make_text_field(title: str, summary: str) -> str:
    # keep it simple: title is high signal
    title = (title or "").strip()
    summary = (summary or "").strip()
    if summary:
        return f"{title}\n{summary}"
    return title
