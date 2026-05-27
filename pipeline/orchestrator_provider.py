"""
FeatureProvider implementation wired to project settings + data ABCs.

Returns pre-computed numpy arrays for ``pipeline.train_orchestrator`` — no
DataFrame manipulation inside the orchestrator itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from adapters.factory import create_data_handler
from config.settings_store import load_settings
from core.interfaces import FeatureRequest
from data.features.pit_features import feature_matrix
from models.xlstm.train import prepare_features
from strategies.bank import build_strategy_bank


def _align_features_to_master(
    bundle_master: pd.DatetimeIndex,
    features: np.ndarray,
    feat_index: pd.DatetimeIndex,
    strategy_matrix: np.ndarray,
    spy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align feature rows to the strategy-bank master timeline."""
    master = pd.DatetimeIndex(bundle_master).normalize()
    feat_index = pd.DatetimeIndex(feat_index).normalize()

    if len(features) != len(feat_index):
        raise ValueError("features and feat_index length mismatch")
    if strategy_matrix.shape[0] != len(master) or spy.shape[0] != len(master):
        raise ValueError("strategy/spy length must match master index")

    master_rows: list[int] = []
    feat_rows: list[int] = []
    for i, ts in enumerate(feat_index):
        loc = master.get_indexer([ts], method=None)
        if loc[0] >= 0:
            master_rows.append(int(loc[0]))
            feat_rows.append(i)

    if len(feat_rows) < 64:
        raise ValueError(f"Insufficient aligned bars ({len(feat_rows)})")

    return (
        features[feat_rows].astype(np.float32),
        strategy_matrix[master_rows].astype(np.float32),
        spy[master_rows].astype(np.float32),
    )


def _load_features_for_symbol(
    settings: dict[str, Any],
    symbol: str,
    primary_bars: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """PiT feature matrix for ``symbol`` via handler or primary bars fallback."""
    feat_cfg = settings["features"]
    try:
        handler = create_data_handler(settings)
        frame = handler.fetch_feature_frame(
            FeatureRequest(
                symbols=[symbol],
                pit_shift=int(feat_cfg["pit_shift"]),
                include_macro=bool(feat_cfg.get("include_macro", False)),
                include_sentiment=bool(feat_cfg.get("include_sentiment", False)),
            )
        )
        if len(frame) < 64:
            raise ValueError("insufficient feature rows from data handler")
        return frame.to_numpy(dtype=np.float64), pd.DatetimeIndex(frame.index)
    except Exception:
        matrix, index = feature_matrix(
            primary_bars,
            pit_shift=int(feat_cfg["pit_shift"]),
            autocorr_window=int(feat_cfg["autocorr_window"]),
            atr_window=int(feat_cfg["atr_window"]),
            frac_diff_d=float(feat_cfg["frac_diff_d"]),
            frac_diff_thresh=float(feat_cfg.get("frac_diff_thresh", 1e-3)),
        )
        return matrix, index


def _synthetic_strategy_matrix(time_steps: int, seed: int = 42) -> np.ndarray:
    """Fallback (T, 3) strategy returns when the bank cannot be built offline."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.001, size=(time_steps, 3)).astype(np.float32)


def _synthetic_spy(time_steps: int, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.0008, size=time_steps).astype(np.float32)


@dataclass
class SettingsFeatureProvider:
    """
    Supplies global and per-ticker arrays from ``settings.yaml`` + adapters.

    Parameters
    ----------
    settings
        Loaded config dict (``load_settings()``).
    use_yfinance
        When False, global features use synthetic OHLCV (``prepare_features``).
        Per-ticker Stage 3 still uses SQLite/yfinance via ``build_strategy_bank``
        when bars exist; otherwise synthetic strategy returns are aligned to
        the feature timeline.
    global_symbol
        Primary symbol for Stage 1 / Stage 2 global training.
    """

    settings: dict[str, Any] = field(default_factory=load_settings)
    use_yfinance: bool = True
    global_symbol: Optional[str] = None

    _global_cache: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default=None, init=False, repr=False
    )
    _ticker_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def primary_symbol(self) -> str:
        return self.global_symbol or self.settings["universe"]["equities"][0]

    def _build_for_primary(self, primary: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bundle = build_strategy_bank(self.settings, primary_symbol=primary)
        features, feat_index = _load_features_for_symbol(
            self.settings, primary, bundle.primary_bars
        )
        strat = bundle.strategy_returns.as_matrix()
        return _align_features_to_master(
            bundle.master_index, features, feat_index, strat, bundle.spy_returns
        )

    def _get_global(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._global_cache is not None:
            return self._global_cache

        if not self.use_yfinance:
            features = prepare_features(
                self.settings, use_yfinance=False
            ).astype(np.float32)
            try:
                feats, strat, spy = self._build_for_primary(self.primary_symbol)
                # Prefer real bank alignment when SQLite has bars; else synthetic legs.
                if feats.shape[0] == features.shape[0]:
                    self._global_cache = (feats, strat, spy)
                else:
                    t = features.shape[0]
                    self._global_cache = (
                        features,
                        _synthetic_strategy_matrix(t),
                        _synthetic_spy(t),
                    )
            except Exception:
                t = features.shape[0]
                self._global_cache = (
                    features,
                    _synthetic_strategy_matrix(t),
                    _synthetic_spy(t),
                )
        else:
            self._global_cache = self._build_for_primary(self.primary_symbol)

        return self._global_cache

    def _get_ticker(self, ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]

        try:
            arrays = self._build_for_primary(ticker)
        except Exception:
            # Last resort: reuse global feature width with synthetic legs on global T.
            feats, _, _ = self._get_global()
            t = feats.shape[0]
            arrays = (
                feats,
                _synthetic_strategy_matrix(t),
                _synthetic_spy(t),
            )
        self._ticker_cache[ticker] = arrays
        return arrays

    def get_global_features(self) -> np.ndarray:
        return self._get_global()[0]

    def get_global_strategy_returns(self) -> np.ndarray:
        return self._get_global()[1]

    def get_global_benchmark_returns(self) -> np.ndarray:
        return self._get_global()[2]

    def get_ticker_features(self, ticker: str) -> np.ndarray:
        return self._get_ticker(ticker)[0]

    def get_ticker_strategy_returns(self, ticker: str) -> np.ndarray:
        return self._get_ticker(ticker)[1]

    def get_ticker_benchmark_returns(self, ticker: str) -> np.ndarray:
        return self._get_ticker(ticker)[2]

    @property
    def input_dim(self) -> int:
        return int(self.get_global_features().shape[1])
