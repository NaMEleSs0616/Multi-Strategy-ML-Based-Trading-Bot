"""Load training artifacts and data snapshots for the Streamlit monitor."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings_store import PROJECT_ROOT, load_settings


@dataclass
class MonitorSnapshot:
    bar_inventory: pd.DataFrame
    price_series: pd.DataFrame
    features: pd.DataFrame
    strategy_returns: pd.DataFrame
    spy_returns: pd.Series
    xlstm_history: Optional[pd.DataFrame]
    ppo_summary: Optional[pd.DataFrame]
    embeddings: Optional[pd.DataFrame]
    checkpoint_status: dict[str, Any]


def _db_path(settings: dict[str, Any]) -> Path:
    return PROJECT_ROOT / settings["data"]["sqlite_path"]


def load_bar_inventory(settings: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    settings = settings or load_settings()
    db = _db_path(settings)
    if not db.exists():
        return pd.DataFrame(columns=["symbol", "interval", "rows", "start", "end"])

    with sqlite3.connect(db) as conn:
        frame = pd.read_sql_query(
            """
            SELECT symbol, interval, COUNT(*) AS rows,
                   MIN(timestamp) AS start, MAX(timestamp) AS end
            FROM bars
            GROUP BY symbol, interval
            ORDER BY symbol, interval
            """,
            conn,
        )
    return frame


def load_price_series(
    symbol: str,
    interval: str = "1d",
    *,
    settings: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    from data.harvester.storage import BarStore

    settings = settings or load_settings()
    store = BarStore(_db_path(settings))
    bars = store.load_bars(symbol, interval)
    if bars.empty:
        return pd.DataFrame(columns=["close"])
    out = bars[["close"]].copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_feature_preview(
    symbol: str,
    *,
    settings: Optional[dict[str, Any]] = None,
    tail: int = 300,
) -> pd.DataFrame:
    from adapters.factory import create_data_handler
    from core.interfaces import FeatureRequest

    settings = settings or load_settings()
    feat_cfg = settings["features"]
    handler = create_data_handler(settings)
    frame = handler.fetch_feature_frame(
        FeatureRequest(
            symbols=[symbol],
            pit_shift=int(feat_cfg["pit_shift"]),
            include_macro=bool(feat_cfg.get("include_macro", False)),
            include_sentiment=bool(feat_cfg.get("include_sentiment", False)),
        )
    )
    if len(frame) > tail:
        frame = frame.iloc[-tail:]
    return frame


def load_strategy_returns_preview(
    *,
    settings: Optional[dict[str, Any]] = None,
    tail: int = 300,
) -> tuple[pd.DataFrame, pd.Series]:
    from pipeline.training_data import build_training_data

    settings = settings or load_settings()
    data = build_training_data(settings, require_encoder=False)
    matrix = data.strategy_returns.as_matrix()
    idx = data.index[-tail:] if len(data.index) > tail else data.index
    start = max(0, len(data.index) - len(idx))
    frame = pd.DataFrame(
        matrix[start:],
        index=idx,
        columns=["stat_arb", "vol_breakout", "mean_reversion"],
    )
    spy = pd.Series(data.spy_returns[start:], index=idx, name="SPY")
    return frame, spy


def load_xlstm_history(settings: Optional[dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    settings = settings or load_settings()
    path = PROJECT_ROOT / settings.get("xlstm", {}).get(
        "checkpoint_dir", "models/xlstm/checkpoints"
    ) / "train_history.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("epoch")


def load_ppo_summary(settings: Optional[dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    settings = settings or load_settings()
    path = PROJECT_ROOT / settings.get("ppo", {}).get(
        "checkpoint_dir", "models/ppo/checkpoints"
    ) / "train_summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_embeddings_preview(
    settings: Optional[dict[str, Any]] = None,
    tail: int = 300,
    dims: int = 3,
) -> Optional[pd.DataFrame]:
    settings = settings or load_settings()
    path = PROJECT_ROOT / settings.get("xlstm", {}).get(
        "checkpoint_dir", "models/xlstm/checkpoints"
    ) / "embeddings.npy"
    if not path.exists():
        return None
    emb = np.load(path)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return None
    if emb.shape[0] > tail:
        emb = emb[-tail:]
    cols = min(dims, emb.shape[1])
    return pd.DataFrame(
        emb[:, :cols],
        columns=[f"emb_{i}" for i in range(cols)],
    )


def checkpoint_status(settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    settings = settings or load_settings()
    xlstm_dir = PROJECT_ROOT / settings.get("xlstm", {}).get(
        "checkpoint_dir", "models/xlstm/checkpoints"
    )
    ppo_dir = PROJECT_ROOT / settings.get("ppo", {}).get(
        "checkpoint_dir", "models/ppo/checkpoints"
    )

    def _stat(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"exists": False, "path": str(path), "mtime": None, "size_kb": 0}
        st = path.stat()
        return {
            "exists": True,
            "path": str(path),
            "mtime": pd.Timestamp(st.st_mtime, unit="s"),
            "size_kb": round(st.st_size / 1024, 1),
        }

    return {
        "xlstm_best": _stat(xlstm_dir / "xlstm_best.pt"),
        "xlstm_frozen": _stat(xlstm_dir / "xlstm_frozen.pt"),
        "embeddings": _stat(xlstm_dir / "embeddings.npy"),
        "train_history": _stat(xlstm_dir / "train_history.json"),
        "ppo_router": _stat(ppo_dir / "ppo_router.zip"),
        "ppo_summary": _stat(ppo_dir / "train_summary.json"),
    }


def build_monitor_snapshot(
    symbol: str,
    *,
    settings: Optional[dict[str, Any]] = None,
) -> MonitorSnapshot:
    settings = settings or load_settings()
    strat, spy = load_strategy_returns_preview(settings=settings)
    return MonitorSnapshot(
        bar_inventory=load_bar_inventory(settings),
        price_series=load_price_series(symbol, settings=settings),
        features=load_feature_preview(symbol, settings=settings),
        strategy_returns=strat,
        spy_returns=spy,
        xlstm_history=load_xlstm_history(settings),
        ppo_summary=load_ppo_summary(settings),
        embeddings=load_embeddings_preview(settings),
        checkpoint_status=checkpoint_status(settings),
    )
