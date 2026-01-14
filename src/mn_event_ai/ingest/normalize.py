from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import dateparser


_ws_re = re.compile(r"\s+")


def clean_text(s: str) -> str:
    import html
    from urllib.parse import unquote_plus

    s = s or ""
    s = s.strip()

    # Decode HTML entities: &amp; &quot; etc.
    s = html.unescape(s)

    # URL decode: %20 -> space, + -> space
    # Safe to apply because RSS often stuffs encoded strings into titles/snippets.
    s = unquote_plus(s)

    # Normalize whitespace
    s = _ws_re.sub(" ", s).strip()
    return s


def canonical_url(url: str) -> str:
    """
    Basic canonicalization:
    - strip whitespace
    - remove obvious tracking query params (utm_*, fbclid, gclid)
    - keep scheme/host/path and remaining query
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    q = parse_qsl(parts.query, keep_blank_values=True)
    q2 = [(k, v) for (k, v) in q if not (k.lower().startswith("utm_") or k.lower() in {"fbclid", "gclid"})]
    query = urlencode(q2)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def parse_dt(value: Any) -> Optional[datetime]:
    """
    feedparser may produce:
      - published_parsed / updated_parsed (time.struct_time)
      - published / updated (string)
    We accept whatever and try to parse.
    """
    if value is None:
        return None
    if hasattr(value, "tm_year"):  # struct_time
        try:
            return datetime(*value[:6], tzinfo=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        dt = dateparser.parse(value, settings={"RETURN_AS_TIMEZONE_AWARE": True})
        if dt is None:
            return None
        return dt.astimezone(timezone.utc)
    return None


def stable_id(source_id: str, url: str, title: str, published_at: Optional[datetime]) -> str:
    base = f"{source_id}|{canonical_url(url)}|{clean_text(title)}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


@dataclass
class NormalizedArticle:
    id: str
    source_id: str
    source_name: str
    url: str
    canonical_url: str
    title: str
    summary: str
    published_at: Optional[str]  # ISO string UTC
    fetched_at_unix: int
    raw_entry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "summary": self.summary,
            "published_at": self.published_at,
            "fetched_at_unix": self.fetched_at_unix,
            "raw_entry": self.raw_entry,
        }


def normalize_entry(
    source_id: str,
    source_name: str,
    fetched_at_unix: int,
    entry: Dict[str, Any],
) -> NormalizedArticle:
    url = entry.get("link") or entry.get("id") or ""
    can_url = canonical_url(url)
    title = clean_text(entry.get("title", ""))
    summary = clean_text(entry.get("summary", "") or entry.get("description", ""))

    # Try best-available timestamp fields
    dt = None
    if "published_parsed" in entry:
        dt = parse_dt(entry.get("published_parsed"))
    if dt is None and "updated_parsed" in entry:
        dt = parse_dt(entry.get("updated_parsed"))
    if dt is None and "published" in entry:
        dt = parse_dt(entry.get("published"))
    if dt is None and "updated" in entry:
        dt = parse_dt(entry.get("updated"))

    published_iso = dt.astimezone(timezone.utc).isoformat() if dt else None
    aid = stable_id(source_id, can_url or url, title, dt)

    return NormalizedArticle(
        id=aid,
        source_id=source_id,
        source_name=source_name,
        url=url,
        canonical_url=can_url,
        title=title,
        summary=summary,
        published_at=published_iso,
        fetched_at_unix=fetched_at_unix,
        raw_entry=entry,
    )
