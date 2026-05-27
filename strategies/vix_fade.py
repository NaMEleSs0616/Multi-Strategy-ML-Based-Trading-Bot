"""
Macro volatility fade (a.k.a. "buy the panic dip").

Uses the FRED ``VIXCLS`` series. The thesis: when implied vol is extremely
elevated **and** has just turned lower (rate-of-change < 0), the panic
phase is subsiding and forward equity returns are historically positive.
The strategy goes maximum long on the underlying asset during those bars
and is flat otherwise.

Signal
------
    position_t = +1  if  VIX_{t-1} > threshold  AND  ΔVIX_{t-1} < 0
    position_t =  0  otherwise

Both the level and the rate-of-change are PiT-shifted (default 1 bar)
before generating the position, so the realized return at bar ``t`` uses
strictly past information.

Graceful degradation
--------------------
When the VIX series is empty (no ``FRED_API_KEY``, FRED rate limit, etc.),
the strategy emits a zero return series — the leg is just inert capital
allocation rather than a hard failure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import pit_shift, position_returns


def vix_fade_returns(
    vix: pd.Series,
    asset_returns: pd.Series,
    *,
    threshold: float = 25.0,
    roc_lookback: int = 1,
    pit_shift_bars: int = 1,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    "Panic subsiding" long signal on the underlying asset.

    Parameters
    ----------
    vix
        Daily VIX level series (e.g. from
        ``data.features.macro_fred.fetch_macro_frame``'s ``vix_level`` column).
    asset_returns
        Daily simple returns of the asset to long when the signal fires
        (typically SPY or the primary equity in the universe). Must be a
        DatetimeIndex-keyed series.
    threshold
        Level above which VIX is considered "panic" (default 25).
    roc_lookback
        Lookback in bars for the VIX rate-of-change (default 1).
    pit_shift_bars
        PiT shift applied to BOTH the VIX level and ROC.
    index
        Optional explicit master index to reindex output onto. If omitted,
        the strategy returns are aligned to ``asset_returns.index``.

    Returns
    -------
    pd.Series
        Strategy returns aligned to ``index`` (or ``asset_returns.index``).
        Missing values are filled with 0.
    """
    out_index = (
        pd.DatetimeIndex(index)
        if index is not None
        else pd.DatetimeIndex(asset_returns.index)
    )

    # Graceful degradation when no VIX is available.
    if vix is None or len(vix) == 0:
        return pd.Series(0.0, index=out_index)

    vix_clean = pd.Series(vix.values, index=pd.DatetimeIndex(vix.index)).astype(float)
    vix_clean = vix_clean.reindex(out_index).ffill(limit=5).fillna(0.0)

    vix_pit = pit_shift(vix_clean, pit_shift_bars)
    vix_roc_pit = pit_shift(
        vix_clean.diff(roc_lookback).fillna(0.0), pit_shift_bars
    )

    signal_long = (vix_pit > float(threshold)) & (vix_roc_pit < 0.0)
    position = pd.Series(0.0, index=out_index)
    position[signal_long] = 1.0

    asset_returns_clean = (
        pd.Series(asset_returns.values, index=pd.DatetimeIndex(asset_returns.index))
        .astype(float)
        .reindex(out_index)
        .fillna(0.0)
    )

    return position_returns(
        position, asset_returns_clean, pit_shift_bars=1
    ).fillna(0.0)


def fetch_vix_series(
    index: pd.DatetimeIndex,
    *,
    pit_shift_bars: int = 0,
) -> pd.Series:
    """
    Convenience helper to pull a PiT-aligned VIX series via the existing
    FRED helper. Returns an **un-shifted** level series by default — the
    strategy itself applies the PiT shift.
    """
    from data.features.macro_fred import fetch_macro_frame

    frame = fetch_macro_frame(index, pit_shift=pit_shift_bars)
    if "vix_level" not in frame.columns:
        return pd.Series(0.0, index=pd.DatetimeIndex(index))
    return frame["vix_level"].astype(float)
