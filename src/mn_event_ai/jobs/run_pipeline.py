from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mn_event_ai.ingest.fetch_rss import run_fetch
from mn_event_ai.ingest.normalize import normalize_entry
from mn_event_ai.ingest.dedupe import dedupe_articles


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

ARTICLES_OUT = OUT_DIR / "articles.jsonl"
DEDUPE_OUT = OUT_DIR / "dedupe_map.json"
REPORT_OUT = OUT_DIR / "ingest_report.json"


def newest_snapshots(raw_dir: Path) -> List[Path]:
    """
    Grab the newest snapshot per source_id.
    Assumes filenames: rss_{source_id}_{YYYYmmdd-HHMMSS}.json
    """
    latest: Dict[str, Path] = {}
    for p in sorted(raw_dir.glob("rss_*.json")):
        name = p.name
        # rss_sourceid_YYYYmmdd-HHMMSS.json
        if not name.startswith("rss_") or not name.endswith(".json"):
            continue
        mid = name[len("rss_") : -len(".json")]
        # split from right once to preserve source_id that might contain underscores
        if "_" not in mid:
            continue
        source_id, _ts = mid.rsplit("_", 1)
        latest[source_id] = p
    return list(latest.values())


def load_snapshot(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = snapshot.get("source", {})
    source_id = src.get("id", "")
    source_name = src.get("name", source_id)
    fetched_at = int(snapshot.get("fetched_at_unix", 0))
    parsed = snapshot.get("parsed", {})
    entries = parsed.get("entries", []) or []

    out: List[Dict[str, Any]] = []
    for e in entries:
        try:
            art = normalize_entry(source_id, source_name, fetched_at, e).to_dict()
            out.append(art)
        except Exception:
            # Skip malformed entries quietly (RSS is messy by nature).
            continue
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def main() -> None:
    # 1) Fetch RSS snapshots
    fetch_results = await run_fetch()

    # 2) Load newest snapshot per source_id
    snaps = newest_snapshots(RAW_DIR)
    all_articles: List[Dict[str, Any]] = []
    per_source_counts: Dict[str, int] = {}

    for sp in snaps:
        snap = load_snapshot(sp)
        src_id = (snap.get("source", {}) or {}).get("id", "unknown")
        arts = normalize_from_snapshot(snap)
        all_articles.extend(arts)
        per_source_counts[src_id] = len(arts)

    # 3) Deduplicate
    unique, dropped_to_kept = dedupe_articles(all_articles)

    # 4) Write outputs
    write_jsonl(ARTICLES_OUT, unique)
    DEDUPE_OUT.write_text(json.dumps(dropped_to_kept, indent=2), encoding="utf-8")
    REPORT_OUT.write_text(
        json.dumps(
            {
                "snapshots_used": [str(p) for p in snaps],
                "fetch_results": fetch_results,
                "normalized_total": len(all_articles),
                "unique_total": len(unique),
                "dropped_total": len(dropped_to_kept),
                "per_source_normalized": per_source_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"normalized_total": len(all_articles), "unique_total": len(unique)}, indent=2))
    print(f"Wrote: {ARTICLES_OUT}")
    print(f"Wrote: {REPORT_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
