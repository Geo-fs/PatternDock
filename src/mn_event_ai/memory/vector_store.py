from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SKLEARN_ERR = e


@dataclass
class VectorStoreConfig:
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)


class VectorStore:
    """
    Tiny TF-IDF cosine vector store.

    Stores:
      - ids: list[str]
      - texts: list[str]
      - meta: list[dict]
    """

    def __init__(self, cfg: VectorStoreConfig = VectorStoreConfig()):
        if TfidfVectorizer is None or cosine_similarity is None:  # pragma: no cover
            raise RuntimeError(
                "scikit-learn is required for VectorStore. Install: pip install scikit-learn\n"
                f"Original import error: {_SKLEARN_ERR!r}"
            )
        self.cfg = cfg
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.meta: List[Dict[str, Any]] = []

        self._vec = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            stop_words="english",
        )
        self._X = None

    def add(self, item_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.ids.append(item_id)
        self.texts.append(text or "")
        self.meta.append(meta or {})
        self._X = None  # invalidate

    def build(self) -> None:
        self._X = self._vec.fit_transform(self.texts)

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.ids:
            return []
        if self._X is None:
            self.build()
        q = self._vec.transform([text or ""])
        sims = cosine_similarity(q, self._X).flatten()
        top = sims.argsort()[::-1][:k]
        out = []
        for i in top:
            out.append(
                {
                    "id": self.ids[int(i)],
                    "score": float(sims[int(i)]),
                    "meta": self.meta[int(i)],
                }
            )
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self.ids, "texts": self.texts, "meta": self.meta, "cfg": self.cfg.__dict__}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "VectorStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = VectorStoreConfig(**payload.get("cfg", {}))
        vs = VectorStore(cfg)
        vs.ids = payload["ids"]
        vs.texts = payload["texts"]
        vs.meta = payload.get("meta", [{} for _ in vs.ids])
        vs._X = None
        return vs
