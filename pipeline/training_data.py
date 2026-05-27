"""Assemble aligned embeddings, strategy returns, and SPY for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings_store import load_settings
from adapters.factory import create_data_handler
from core.interfaces import FeatureRequest
from data.features.pit_features import feature_matrix
from models.xlstm.inference import encode_feature_matrix, load_frozen_encoder
from rl.gym_trading_env import StrategyReturns
from strategies.bank import MarketBundle, build_strategy_bank


@dataclass
class TrainingData:
    embeddings: np.ndarray
    strategy_returns: StrategyReturns
    spy_returns: np.ndarray
    index: pd.DatetimeIndex
    features: np.ndarray


def _normalize_daily_index(
    idx: pd.DatetimeIndex,
    *,
    tz: str = "America/New_York",
    dedupe: bool = False,
) -> pd.DatetimeIndex:
    """
    Normalize index to timezone-naive daily stamps for robust joins.

    Feature frames may arrive tz-aware (Alpaca) while strategy-bank indices are
    often tz-naive. We convert aware indices to a common market timezone before
    dropping tz, then normalize to calendar day.
    """
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert(tz)
        out = out.tz_localize(None)
    out = out.normalize()
    if dedupe and out.has_duplicates:
        out = pd.DatetimeIndex(out[~out.duplicated(keep="last")])
    return out


def align_to_feature_index(
    bundle: MarketBundle,
    features: np.ndarray,
    feat_index: pd.DatetimeIndex,
    embeddings: np.ndarray,
) -> TrainingData:
    master = _normalize_daily_index(pd.DatetimeIndex(bundle.master_index), dedupe=True)
    feat_index = _normalize_daily_index(pd.DatetimeIndex(feat_index), dedupe=False)

    if len(features) != len(feat_index) or len(embeddings) != len(features):
        raise ValueError("features, index, and embeddings must have equal length")

    # Fast date-map join; resilient to index metadata drift.
    master_lookup: dict[pd.Timestamp, int] = {}
    for i, ts in enumerate(master):
        master_lookup[ts] = i

    master_rows: list[int] = []
    feat_rows: list[int] = []
    for i, ts in enumerate(feat_index):
        loc = master_lookup.get(ts)
        if loc is not None:
            master_rows.append(loc)
            feat_rows.append(i)

    if len(feat_rows) < 64:
        raise ValueError(f"Insufficient aligned bars ({len(feat_rows)}) for training")

    return TrainingData(
        embeddings=embeddings[feat_rows].astype(np.float64),
        strategy_returns=bundle.strategy_returns.slice(np.asarray(master_rows)),
        spy_returns=bundle.spy_returns[master_rows].astype(np.float64),
        index=feat_index[feat_rows],
        features=features[feat_rows],
    )


def build_training_data(
    settings: Optional[dict[str, Any]] = None,
    *,
    require_encoder: bool = False,
) -> TrainingData:
    settings = settings or load_settings()
    bundle = build_strategy_bank(settings)
    feat_cfg = settings["features"]

    try:
        handler = create_data_handler(settings)
        primary = settings["universe"]["equities"][0]
        frame = handler.fetch_feature_frame(
            FeatureRequest(
                symbols=[primary],
                pit_shift=int(feat_cfg["pit_shift"]),
                include_macro=bool(feat_cfg.get("include_macro", False)),
                include_sentiment=bool(feat_cfg.get("include_sentiment", False)),
            )
        )
        if len(frame) < 64:
            raise ValueError("insufficient feature rows from data handler")
        features = frame.to_numpy(dtype=np.float64)
        feat_index = pd.DatetimeIndex(frame.index)
    except Exception:
        features, feat_index = feature_matrix(
            bundle.primary_bars,
            pit_shift=int(feat_cfg["pit_shift"]),
            autocorr_window=int(feat_cfg["autocorr_window"]),
            atr_window=int(feat_cfg["atr_window"]),
            frac_diff_d=float(feat_cfg["frac_diff_d"]),
            frac_diff_thresh=float(feat_cfg.get("frac_diff_thresh", 1e-3)),
        )

    try:
        encoder = load_frozen_encoder(settings=settings)
        embeddings = encode_feature_matrix(encoder, features)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        if require_encoder:
            raise
        dim = int(settings["rl"]["embedding_dim"])
        if features.shape[1] < dim:
            embeddings = np.hstack(
                [features, np.zeros((features.shape[0], dim - features.shape[1]))]
            )
        else:
            embeddings = features[:, :dim]

    return align_to_feature_index(bundle, features, feat_index, embeddings)
