from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mn_event_ai.memory.vector_store import VectorStore


@dataclass
class RetrievalResult:
    items: List[Dict[str, Any]]


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        return RetrievalResult(items=self.store.query(query, k=k))
