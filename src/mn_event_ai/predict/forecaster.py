from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from mn_event_ai.predict.drn import DRN


INCIDENTS_IN = Path("data/processed/incidents.jsonl")
ARTICLES_IN = Path("data/processed/articles.jsonl")

FORECAST_OUT = Path("data/processed/incident_forecasts.jsonl")
MODEL_OUT = Path("models/drn.pt")


@dataclass
class ForecastConfig:
    bin_minutes: int = 60
    window_bins: int = 12  # lookback window (e.g., 12 hours if bin=60)
    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_time(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_article_time_map() -> Dict[str, Optional[datetime]]:
    articles = _load_jsonl(ARTICLES_IN)
    return {a["id"]: _parse_time(a.get("published_at")) for a in articles}


def _incident_series(article_times: List[datetime], cfg: ForecastConfig) -> List[float]:
    """
    Build a time-binned count series for a single incident.
    """
    if not article_times:
        return []

    t0 = min(article_times)
    t1 = max(article_times)
    bin_td = timedelta(minutes=cfg.bin_minutes)

    # Extend one bin past last observation (for next-step prediction targets)
    num_bins = int((t1 - t0) / bin_td) + 2
    series = [0.0] * num_bins

    for t in article_times:
        idx = int((t - t0) / bin_td)
        if 0 <= idx < len(series):
            series[idx] += 1.0

    return series


def _make_windows(series: List[float], window: int) -> List[Tuple[List[float], float]]:
    """
    Sliding windows: X[t-window:t] -> y[t]
    """
    out = []
    if len(series) <= window:
        return out
    for t in range(window, len(series)):
        x = series[t - window : t]
        y = series[t]
        out.append((x, y))
    return out


def train_drn(cfg: ForecastConfig = ForecastConfig()) -> DRN:
    incs = _load_jsonl(INCIDENTS_IN)
    tmap = _build_article_time_map()

    samples: List[Tuple[List[float], float]] = []
    for inc in incs:
        times = [tmap.get(aid) for aid in inc["article_ids"]]
        times = [t for t in times if t is not None]
        if len(times) < 2:
            continue
        series = _incident_series(times, cfg)
        samples.extend(_make_windows(series, cfg.window_bins))

    if not samples:
        raise RuntimeError("Not enough incident history to train DRN yet. Ingest more data first.")

    # Create tensors
    X = torch.tensor([s[0] for s in samples], dtype=torch.float32).unsqueeze(-1)  # [N, T, 1]
    y = torch.tensor([s[1] for s in samples], dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    # Shuffle + split
    idx = torch.randperm(X.size(0))
    X, y = X[idx], y[idx]

    split = int(0.8 * X.size(0))
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]

    model = DRN(input_dim=1, hidden_dim=64).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    def run_epoch(train: bool):
        model.train(train)
        total = 0.0
        n = 0
        Xuse, yuse = (Xtr, ytr) if train else (Xva, yva)
        for i in range(0, Xuse.size(0), cfg.batch_size):
            xb = Xuse[i : i + cfg.batch_size].to(cfg.device)
            yb = yuse[i : i + cfg.batch_size].to(cfg.device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(True)
        va = run_epoch(False)
        print(json.dumps({"epoch": ep, "train_mse": tr, "val_mse": va}))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    return model


@torch.no_grad()
def forecast_incidents(cfg: ForecastConfig = ForecastConfig()) -> None:
    # Train (or later: load from disk)
    model = train_drn(cfg)
    model.eval()

    incs = _load_jsonl(INCIDENTS_IN)
    tmap = _build_article_time_map()

    out_rows = []
    for inc in incs:
        times = [tmap.get(aid) for aid in inc["article_ids"]]
        times = [t for t in times if t is not None]
        if len(times) < 2:
            continue

        series = _incident_series(times, cfg)
        if len(series) < cfg.window_bins:
            continue
        x = torch.tensor(series[-cfg.window_bins:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(cfg.device)
        pred = model(x).squeeze().item()

        out_rows.append(
            {
                "incident_id": inc["incident_id"],
                "pred_next_bin_intensity": float(pred),
                "bin_minutes": cfg.bin_minutes,
                "window_bins": cfg.window_bins,
                "summary": inc.get("summary", {}),
            }
        )

    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with FORECAST_OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"wrote": str(FORECAST_OUT), "forecasts": len(out_rows)}, indent=2))


if __name__ == "__main__":
    forecast_incidents()
