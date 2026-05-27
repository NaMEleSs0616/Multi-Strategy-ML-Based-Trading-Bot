"""Shared PiT helpers for strategy return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pit_shift(series: pd.Series, bars: int = 1) -> pd.Series:
    if bars <= 0:
        return series
    return series.shift(bars)


def position_returns(
    position: pd.Series,
    asset_returns: pd.Series,
    *,
    pit_shift_bars: int = 1,
) -> pd.Series:
    """
    Strategy return at t = position_{t-1} * asset_return_t.

    ``position`` must already be computed from PiT-shifted indicators.
    """
    pos = pit_shift(position.fillna(0.0), pit_shift_bars)
    return pos * asset_returns


def compound_to_daily(
    returns: pd.Series,
    master_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compound intraday simple returns onto a daily master calendar."""
    if returns.empty:
        return pd.Series(0.0, index=master_index)

    idx = pd.to_datetime(returns.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame = pd.DataFrame({"r": returns.values}, index=idx)
    daily = (1.0 + frame["r"]).groupby(pd.Grouper(freq="D")).prod() - 1.0
    daily.index = pd.to_datetime(daily.index).normalize()
    master = pd.to_datetime(master_index).normalize()
    aligned = daily.reindex(master, fill_value=0.0)
    return aligned.astype(float)


def align_returns(series: pd.Series, master_index: pd.DatetimeIndex) -> np.ndarray:
    """Reindex returns to master (daily) timeline, fill missing with 0."""
    if isinstance(series.index, pd.DatetimeIndex):
        s = series.copy()
        s.index = pd.to_datetime(s.index).normalize()
        master = pd.to_datetime(master_index).normalize()
        aligned = s.reindex(master, fill_value=0.0)
        return aligned.to_numpy(dtype=np.float64)
    return compound_to_daily(series, master_index).to_numpy(dtype=np.float64)
