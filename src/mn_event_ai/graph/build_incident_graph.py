from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional dependency: scikit-learn
# If missing, you get a clear error instead of silent nonsense.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SKLEARN_ERR = e


@dataclass
class GraphConfig:
    time_window_hours: int = 48
    min_similarity: float = 0.25
    max_neighbors: int = 8  # cap edges per node to avoid quadratic explosion
    text_field: str = "text"  # from weak labels / or build from title+summary externally


def _parse_time(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept ISO strings
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def build_edges(
    rows: List[Dict],
    cfg: GraphConfig = GraphConfig(),
) -> List[Dict]:
    """
    Returns edges as dicts:
    { "src": idA, "dst": idB, "sim": float, "dt_hours": float }
    """
    if TfidfVectorizer is None or cosine_similarity is None:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for incident graph building. "
            "Install it: pip install scikit-learn\n"
            f"Original import error: {_SKLEARN_ERR!r}"
        )

    texts = []
    times: List[Optional[datetime]] = []
    ids: List[str] = []

    for r in rows:
        ids.append(r["id"])
        texts.append((r.get(cfg.text_field) or "").strip())
        times.append(_parse_time(r.get("published_at")))

    vec = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    S = cosine_similarity(X)

    # Build candidate edges within time window and above similarity threshold.
    edges: List[Dict] = []
    n = len(ids)
    win = cfg.time_window_hours

    for i in range(n):
        # Sort neighbors by similarity descending (excluding self)
        sims = [(j, float(S[i, j])) for j in range(n) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)

        kept = 0
        for j, sim in sims:
            if sim < cfg.min_similarity:
                break

            ti, tj = times[i], times[j]
            if ti and tj:
                dt_h = abs((ti - tj).total_seconds()) / 3600.0
                if dt_h > win:
                    continue
            else:
                dt_h = None  # unknown

            edges.append(
                {
                    "src": ids[i],
                    "dst": ids[j],
                    "sim": sim,
                    "dt_hours": dt_h,
                }
            )
            kept += 1
            if kept >= cfg.max_neighbors:
                break

    # De-duplicate undirected edges by keeping max(sim) for (min, max)
    best: Dict[Tuple[str, str], Dict] = {}
    for e in edges:
        a, b = e["src"], e["dst"]
        key = (a, b) if a < b else (b, a)
        if key not in best or e["sim"] > best[key]["sim"]:
            best[key] = e

    return list(best.values())


def load_predictions(pred_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_edges(edges: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
