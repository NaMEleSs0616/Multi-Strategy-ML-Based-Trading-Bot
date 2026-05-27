"""
Point-in-time (PiT) OHLCV feature matrix for xLSTM pre-training.

All rolling statistics are shifted by ``pit_shift`` (default 1) before use.
Unbounded forward-fill is not applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.features.fracdiff import fractional_diff


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
    """Lag-1 autocorrelation in a rolling window."""

    def _acf(x: np.ndarray) -> float:
        if len(x) < 3 or np.std(x) < 1e-12:
            return 0.0
        a = x[:-1]
        b = x[1:]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return series.rolling(window, min_periods=window).apply(_acf, raw=True)


def build_pit_feature_frame(
    bars: pd.DataFrame,
    *,
    pit_shift: int = 1,
    autocorr_window: int = 20,
    atr_window: int = 14,
    frac_diff_d: float = 0.4,
    frac_diff_thresh: float = 1e-3,
) -> pd.DataFrame:
    """
    Build PiT-safe features from OHLCV bars.

    Columns: log_return, frac_diff, autocorr, atr_norm, rel_volume
    """
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower(): c for c in bars.columns}
    frame = bars.rename(columns={cols[k]: k for k in required if k in cols}).copy()
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")

    close = frame["close"].astype(float)
    log_return = np.log(close / close.shift(1))

    log_close = np.log(close.replace(0, np.nan))
    frac_diff = fractional_diff(log_close, frac_diff_d, threshold=frac_diff_thresh)

    autocorr = rolling_autocorr(close, autocorr_window)
    tr = _true_range(frame["high"], frame["low"], close)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    atr_norm = atr / close.replace(0, np.nan)

    vol_mean = frame["volume"].rolling(20, min_periods=20).mean()
    rel_volume = frame["volume"] / vol_mean.replace(0, np.nan)

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "frac_diff": frac_diff,
            "autocorr": autocorr,
            "atr_norm": atr_norm,
            "rel_volume": rel_volume,
        },
        index=frame.index,
    )

    if pit_shift > 0:
        features = features.shift(pit_shift)

    return features.dropna(how="any")


def feature_matrix(bars: pd.DataFrame, **kwargs) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (T, F) float matrix and aligned index."""
    frame = build_pit_feature_frame(bars, **kwargs)
    return frame.to_numpy(dtype=np.float64), frame.index
