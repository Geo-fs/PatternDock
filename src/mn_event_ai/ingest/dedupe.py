from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def content_fingerprint(title: str, summary: str) -> str:
    """
    Fingerprint for aggressive dedupe across sources.
    """
    t = (title or "").strip().lower()
    s = (summary or "").strip().lower()
    # Truncate summary to reduce long boilerplate dominance
    s = s[:500]
    return _sha(t + "||" + s)


def is_near_duplicate(a_title: str, b_title: str, max_norm_lev: float = 0.12) -> bool:
    """
    Near-dup check on titles. Normalized Levenshtein distance.
    Lower is closer.
    """
    a = (a_title or "").strip().lower()
    b = (b_title or "").strip().lower()
    if not a or not b:
        return False
    dist = Levenshtein.distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return False
    return (dist / denom) <= max_norm_lev


def dedupe_articles(articles: Iterable[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Input: iterable of normalized article dicts (must have id, canonical_url, title, summary)
    Output:
      - unique articles
      - map: dropped_id -> kept_id
    Strategy:
      1) canonical_url exact match
      2) content fingerprint match
      3) near-dup title match within a small rolling window (O(n^2) worst case; keep batches modest)
    """
    uniques: List[Dict] = []
    dropped_to_kept: Dict[str, str] = {}

    seen_url: Dict[str, str] = {}
    seen_fp: Dict[str, str] = {}

    for art in articles:
        aid = art["id"]
        can = (art.get("canonical_url") or "").strip()
        title = art.get("title") or ""
        summary = art.get("summary") or ""

        if can and can in seen_url:
            dropped_to_kept[aid] = seen_url[can]
            continue

        fp = content_fingerprint(title, summary)
        if fp in seen_fp:
            dropped_to_kept[aid] = seen_fp[fp]
            continue

        # Near-dup title vs last ~200 uniques (cheap-ish)
        kept = None
        for prev in uniques[-200:]:
            if is_near_duplicate(title, prev.get("title", "")):
                kept = prev["id"]
                break
        if kept:
            dropped_to_kept[aid] = kept
            continue

        uniques.append(art)
        if can:
            seen_url[can] = aid
        seen_fp[fp] = aid

    return uniques, dropped_to_kept
