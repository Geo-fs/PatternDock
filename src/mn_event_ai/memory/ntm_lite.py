from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from mn_event_ai.memory.vector_store import VectorStore
from mn_event_ai.memory.retrieval import Retriever


@dataclass
class MemoryItem:
    id: str
    text: str
    meta: Dict[str, Any]


class NTMLite:
    """
    NTM-lite:
      - short-term memory: last N items (deque)
      - long-term memory: vector store for retrieval
    """

    def __init__(self, store: VectorStore, short_capacity: int = 50):
        self.short: Deque[MemoryItem] = deque(maxlen=short_capacity)
        self.store = store
        self.retriever = Retriever(store)

    def write(self, item_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        mi = MemoryItem(id=item_id, text=text or "", meta=meta or {})
        self.short.appendleft(mi)
        self.store.add(item_id, mi.text, mi.meta)

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        out = []
        for mi in list(self.short)[:n]:
            out.append({"id": mi.id, "text": mi.text, "meta": mi.meta})
        return out

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.search(query, k=k).items
