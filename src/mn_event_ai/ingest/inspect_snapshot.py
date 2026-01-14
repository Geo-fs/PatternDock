from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def newest_snapshot(raw_dir: Path, source_id: str) -> Optional[Path]:
    files = sorted(raw_dir.glob(f"rss_{source_id}_*.json"))
    return files[-1] if files else None


def inspect_file(path: Path, max_head: int = 400) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    src = data.get("source", {})
    parsed = data.get("parsed", {})
    http = data.get("http", {})

    print("=== SOURCE ===")
    print(f"id: {src.get('id')}")
    print(f"name: {src.get('name')}")
    print(f"url: {src.get('url')}")
    print()

    print("=== HTTP ===")
    print(f"status_code: {http.get('status_code')}")
    print(f"final_url:   {http.get('final_url')}")
    print(f"content_type:{http.get('content_type')}")
    print(f"used_jina:   {http.get('used_jina')}")
    print(f"is_html:     {data.get('content_is_html')}")
    print(f"content_len: {data.get('content_len')}")
    print()

    print("=== FEEDPARSER ===")
    print(f"bozo: {parsed.get('bozo')}")
    be = parsed.get("bozo_exception", "")
    if be:
        print(f"bozo_exception: {be[:200]}")
    print(f"feed keys: {list((parsed.get('feed') or {}).keys())[:20]}")
    print(f"entries: {len(parsed.get('entries') or [])}")
    print()

    # Show sample entry fields if present
    entries = parsed.get("entries") or []
    if entries:
        e0 = entries[0]
        print("=== SAMPLE ENTRY KEYS ===")
        print(sorted(list(e0.keys()))[:40])
        print()
        print("=== SAMPLE ENTRY TITLE/LINK ===")
        print("title:", e0.get("title", "")[:200])
        print("link: ", e0.get("link", "")[:200])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source_id from feeds.yaml (e.g., nws_mpx_rss)")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    snap = newest_snapshot(raw_dir, args.source)
    if not snap:
        raise SystemExit(f"No snapshots found for source '{args.source}' in {raw_dir}")

    inspect_file(snap)


if __name__ == "__main__":
    main()
