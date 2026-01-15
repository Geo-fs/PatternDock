from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query

from mn_event_ai.jobs.run_pipeline import main as run_pipeline_main  # type: ignore
from mn_event_ai.graph.cluster_incidents import main as cluster_main

from mn_event_ai.memory.vector_store import VectorStore
from mn_event_ai.memory.ntm_lite import NTMLite

ARTICLES = Path("data/processed/articles.jsonl")
PREDICTIONS = Path("data/processed/article_predictions.jsonl")
INCIDENTS = Path("data/processed/incidents.jsonl")


app = FastAPI(title="mn-event-ai (PatternDock)", version="0.1.0")

# Lazy-initialized memory
_mem: Optional[NTMLite] = None


def _load_jsonl(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ensure_memory() -> NTMLite:
    global _mem
    if _mem is not None:
        return _mem
    store = VectorStore()
    mem = NTMLite(store)

    # Seed with recent articles
    for r in _load_jsonl(ARTICLES, limit=500):
        text = f"{r.get('title','')}\n{r.get('summary','')}".strip()
        mem.write(r["id"], text, meta={"source_id": r.get("source_id"), "url": r.get("canonical_url")})
    _mem = mem
    return _mem


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "articles": ARTICLES.exists(),
        "predictions": PREDICTIONS.exists(),
        "incidents": INCIDENTS.exists(),
    }


@app.post("/run/pipeline")
def run_pipeline() -> Dict[str, Any]:
    # Runs the pipeline (fetch → normalize → dedupe).
    # NOTE: This is synchronous. In production you'd background-task it.
    run_pipeline_main()  # type: ignore

    # If predictions exist (user ran inference), cluster incidents.
    if PREDICTIONS.exists():
        cluster_main()

    # Refresh memory cache
    global _mem
    _mem = None

    return {"ok": True}


@app.get("/articles")
def get_articles(limit: int = 50) -> List[Dict[str, Any]]:
    return _load_jsonl(ARTICLES, limit=limit)


@app.get("/incidents")
def get_incidents(limit: int = 50) -> List[Dict[str, Any]]:
    return _load_jsonl(INCIDENTS, limit=limit)


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 5) -> Dict[str, Any]:
    mem = _ensure_memory()
    hits = mem.retrieve(q, k=k)
    return {"query": q, "hits": hits}
