from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import httpx
import yaml

RAW_DIR_DEFAULT = Path("data/raw")
CONFIG_DEFAULT = Path("configs/feeds.yaml")


@dataclass
class FeedSource:
    id: str
    name: str
    url: str
    enabled: bool = True
    tags: List[str] = None
    timeout_seconds: int = 20
    max_items_per_pull: int = 100


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sources(config_path: Path = CONFIG_DEFAULT) -> Tuple[Dict[str, Any], List[FeedSource]]:
    cfg = _load_yaml(config_path)
    defaults = cfg.get("defaults", {})
    sources_cfg = cfg.get("sources", [])

    sources: List[FeedSource] = []
    for s in sources_cfg:
        enabled = bool(s.get("enabled", defaults.get("enabled", True)))
        if not enabled:
            continue

        sources.append(
            FeedSource(
                id=str(s["id"]),
                name=str(s.get("name", s["id"])),
                url=str(s["url"]),
                enabled=enabled,
                tags=list(s.get("tags", [])),
                timeout_seconds=int(s.get("timeout_seconds", defaults.get("timeout_seconds", 20))),
                max_items_per_pull=int(s.get("max_items_per_pull", defaults.get("max_items_per_pull", 100))),
            )
        )

    return cfg, sources


def _ua_from_env(cfg: Dict[str, Any]) -> str:
    defaults = cfg.get("defaults", {})
    ua_env = defaults.get("user_agent_env", "MN_EVENT_AI_USER_AGENT")
    ua = os.getenv(ua_env)
    if ua:
        return ua
    # Fallback UA that looks less like a bot wrote it at 3am.
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 mn-event-ai/0.1"
    )


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _looks_like_html(content: bytes, content_type: str) -> bool:
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return True
    sniff = content[:400].lstrip().lower()
    return sniff.startswith(b"<!doctype html") or sniff.startswith(b"<html") or b"<head" in sniff


def _jina_proxy_url(original: str) -> str:
    # Jina expects r.jina.ai/http://example.com/path (or https)
    clean = original.strip()
    if clean.startswith("https://"):
        clean = "https://" + clean[len("https://") :]
        return "https://r.jina.ai/http://" + clean[len("https://") :]
    if clean.startswith("http://"):
        clean = "http://" + clean[len("http://") :]
        return "https://r.jina.ai/http://" + clean[len("http://") :]
    return "https://r.jina.ai/http://" + clean


async def fetch_feed_xml(client: httpx.AsyncClient, source: FeedSource) -> Tuple[bytes, Dict[str, Any]]:
    """
    Returns (content_bytes, meta)
    meta contains status code, final url, content-type, and whether jina fallback was used.
    """
    meta: Dict[str, Any] = {"used_jina": False, "status_code": None, "final_url": source.url, "content_type": None}

    backoff = 1.25
    last_exc: Optional[Exception] = None

    for attempt in range(1, 5):  # 4 attempts
        try:
            r = await client.get(source.url, timeout=source.timeout_seconds)
            meta["status_code"] = r.status_code
            meta["final_url"] = str(r.url)
            meta["content_type"] = r.headers.get("content-type", "")

            # If blocked, try Jina proxy once per attempt cycle
            if r.status_code == 403:
                alt = _jina_proxy_url(source.url)
                r2 = await client.get(alt, timeout=source.timeout_seconds)
                meta["used_jina"] = True
                meta["status_code"] = r2.status_code
                meta["final_url"] = str(r2.url)
                meta["content_type"] = r2.headers.get("content-type", "")
                r2.raise_for_status()
                return r2.content, meta

            r.raise_for_status()
            return r.content, meta

        except httpx.HTTPStatusError as e:
            last_exc = e
            code = e.response.status_code if e.response is not None else None
            # Backoff on common transient / block / rate-limit codes
            if code in (403, 429, 500, 502, 503, 504):
                await asyncio.sleep(backoff * attempt)
                continue
            raise
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
            last_exc = e
            await asyncio.sleep(backoff * attempt)
            continue
        except Exception as e:
            last_exc = e
            await asyncio.sleep(backoff * attempt)
            continue

    raise last_exc if last_exc else RuntimeError("Unknown fetch error")


def parse_feed(xml_bytes: bytes) -> Dict[str, Any]:
    parsed = feedparser.parse(xml_bytes)
    return {
        "bozo": bool(getattr(parsed, "bozo", False)),
        "bozo_exception": str(getattr(parsed, "bozo_exception", "")) if getattr(parsed, "bozo", False) else "",
        "feed": dict(getattr(parsed, "feed", {})),
        "entries": [dict(e) for e in getattr(parsed, "entries", [])],
    }


def write_raw_snapshot(raw_dir: Path, source_id: str, payload: Dict[str, Any]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = raw_dir / f"rss_{source_id}_{ts}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


async def fetch_and_store_one(
    client: httpx.AsyncClient,
    raw_dir: Path,
    source: FeedSource,
) -> Dict[str, Any]:
    content, meta = await fetch_feed_xml(client, source)

    # If the site returned HTML (common when blocked), record it explicitly.
    is_html = _looks_like_html(content, meta.get("content_type", ""))

    parsed = parse_feed(content) if not is_html else {
        "bozo": True,
        "bozo_exception": "Received HTML instead of RSS/XML (likely blocked or redirected).",
        "feed": {},
        "entries": [],
    }

    # Respect max_items_per_pull if feed is huge
    entries = parsed.get("entries", [])
    if isinstance(entries, list) and len(entries) > source.max_items_per_pull:
        parsed["entries"] = entries[: source.max_items_per_pull]

    snapshot = {
        "source": {"id": source.id, "name": source.name, "url": source.url, "tags": source.tags},
        "fetched_at_unix": int(time.time()),
        "http": meta,
        "content_sha256": _hash_bytes(content),
        "content_len": len(content),
        "content_is_html": is_html,
        "parsed": parsed,
    }
    path = write_raw_snapshot(raw_dir, source.id, snapshot)

    return {
        "source_id": source.id,
        "snapshot_path": str(path),
        "entries": len(parsed.get("entries", [])),
        "used_jina": bool(meta.get("used_jina", False)),
        "status_code": meta.get("status_code"),
        "content_is_html": is_html,
    }


async def run_fetch(config_path: Path = CONFIG_DEFAULT, raw_dir: Path = RAW_DIR_DEFAULT) -> List[Dict[str, Any]]:
    cfg, sources = load_sources(config_path)
    ua = _ua_from_env(cfg)

    # More browser-like headers reduce blocks.
    headers = {
        "User-Agent": ua,
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(headers=headers, limits=limits, follow_redirects=True) as client:
        tasks = [fetch_and_store_one(client, raw_dir, s) for s in sources]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for s, r in zip(sources, gathered):
            if isinstance(r, Exception):
                results.append({"source_id": s.id, "error": repr(r)})
            else:
                results.append(r)

    # Print a tiny health summary so you can see if you're winning.
    ok = sum(1 for r in results if "error" not in r)
    blocked = sum(1 for r in results if r.get("status_code") == 403 or r.get("content_is_html") is True)
    zero = sum(1 for r in results if "error" not in r and r.get("entries", 0) == 0)
    print(json.dumps({"sources": len(results), "ok": ok, "blocked_or_html": blocked, "zero_entries": zero}, indent=2))

    return results


if __name__ == "__main__":
    out = asyncio.run(run_fetch())
    print(json.dumps(out, indent=2))
