from __future__ import annotations

import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from mn_event_ai.graph.build_incident_graph import GraphConfig, build_edges


PRED_IN_DEFAULT = Path("data/processed/article_predictions.jsonl")
INCIDENTS_OUT_DEFAULT = Path("data/processed/incidents.jsonl")
EDGES_OUT_DEFAULT = Path("data/processed/incident_edges.jsonl")


@dataclass
class ClusterConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    min_cluster_size: int = 2

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _build_adj(edges: List[Dict]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        a, b = e["src"], e["dst"]
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _connected_components(nodes: List[str], adj: Dict[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    comps: List[List[str]] = []

    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp = []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def _incident_id(i: int) -> str:
    return f"inc_{i:06d}"


def _summarize_incident(rows_by_id: Dict[str, Dict], article_ids: List[str]) -> Dict:
    # pick a representative title and label distribution
    titles = []
    label_counter = Counter()
    urls = []
    times = []

    for aid in article_ids:
        r = rows_by_id[aid]
        t = (r.get("title") or "").strip()
        if t:
            titles.append(t)
        urls.append(r.get("canonical_url"))
        times.append(r.get("published_at"))
        for lab in r.get("pred_labels", []) or []:
            label_counter[lab] += 1

    top_title = titles[0] if titles else ""
    top_labels = [lab for lab, _n in label_counter.most_common(3)]

    return {
        "size": len(article_ids),
        "title_hint": top_title,
        "top_labels": top_labels,
        "label_hist": dict(label_counter),
        "published_min": min([t for t in times if t] or [None]),
        "published_max": max([t for t in times if t] or [None]),
    }


def cluster_incidents(
    pred_rows: List[Dict],
    cfg: ClusterConfig = ClusterConfig(),
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (incidents, edges).
    incidents: list of dicts:
      {
        "incident_id": "...",
        "article_ids": [...],
        "summary": {...}
      }
    edges: graph edges used for clustering
    """
    from pathlib import Path
    import json

    # Join summaries from articles.jsonl so clustering uses richer text
    articles_path = Path("data/processed/articles.jsonl")
    a_map = {}
    if articles_path.exists():
        with articles_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    a_map[a["id"]] = a

    rows = []
    for r in pred_rows:
        a = a_map.get(r["id"], {})
        title = (r.get("title") or a.get("title") or "").strip()
        summary = (a.get("summary") or "").strip()
        text = f"{title}\n{summary}".strip()

        r2 = dict(r)
        r2["text"] = text
        rows.append(r2)


    edges = build_edges(rows, cfg.graph)
    adj = _build_adj(edges)
    ids = [r["id"] for r in rows]
    rows_by_id = {r["id"]: r for r in rows}

    comps = _connected_components(ids, adj)

    incidents: List[Dict] = []
    inc_i = 0

    for comp in comps:
        if len(comp) < cfg.min_cluster_size:
            # singleton incidents
            for aid in comp:
                inc = {
                    "incident_id": _incident_id(inc_i),
                    "article_ids": [aid],
                    "summary": _summarize_incident(rows_by_id, [aid]),
                }
                incidents.append(inc)
                inc_i += 1
        else:
            inc = {
                "incident_id": _incident_id(inc_i),
                "article_ids": sorted(comp),
                "summary": _summarize_incident(rows_by_id, sorted(comp)),
            }
            incidents.append(inc)
            inc_i += 1

    return incidents, edges


def main(
    pred_in: Path = PRED_IN_DEFAULT,
    incidents_out: Path = INCIDENTS_OUT_DEFAULT,
    edges_out: Path = EDGES_OUT_DEFAULT,
) -> None:
    pred_rows = _load_jsonl(pred_in)
    incidents, edges = cluster_incidents(pred_rows)

    incidents_out.parent.mkdir(parents=True, exist_ok=True)
    with incidents_out.open("w", encoding="utf-8") as f:
        for inc in incidents:
            f.write(json.dumps(inc, ensure_ascii=False) + "\n")

    with edges_out.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(json.dumps({"incidents": len(incidents), "edges": len(edges), "wrote": str(incidents_out)}, indent=2))


if __name__ == "__main__":
    main()

