"""
Macro features via FRED (10Y–2Y spread, VIX) with graceful fallback.

All series are PiT-shifted by ``pit_shift`` before return.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs
SERIES_10Y = "DGS10"
SERIES_2Y = "DGS2"
SERIES_VIX = "VIXCLS"


def _fred_available() -> bool:
    return bool(os.environ.get("FRED_API_KEY", "").strip())


def _fetch_fred_series(series_id: str, start: Optional[str] = None) -> pd.Series:
    from fredapi import Fred

    api_key = os.environ["FRED_API_KEY"]
    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start)
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(series.values, index=idx.normalize(), dtype=float)
    return out.sort_index()


def fetch_macro_frame(
    index: pd.DatetimeIndex,
    *,
    pit_shift: int = 1,
    start: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build macro columns aligned to ``index``.

    Columns: ``yield_spread_10y2y``, ``vix_level``.
    Without ``FRED_API_KEY``, returns zeros (PiT-shifted).
    """
    idx = pd.DatetimeIndex(index).normalize()
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    empty = pd.DataFrame(
        {"yield_spread_10y2y": 0.0, "vix_level": 0.0},
        index=idx,
    )

    if not _fred_available():
        logger.info("FRED_API_KEY not set — macro features default to 0")
        if pit_shift > 0:
            empty = empty.shift(pit_shift)
        return empty.reindex(idx).fillna(0.0)

    try:
        y10 = _fetch_fred_series(SERIES_10Y, start=start)
        y2 = _fetch_fred_series(SERIES_2Y, start=start)
        vix = _fetch_fred_series(SERIES_VIX, start=start)
        spread = (y10 - y2).rename("yield_spread_10y2y")
        vix = vix.rename("vix_level")
        macro = pd.concat([spread, vix], axis=1)
        macro = macro.reindex(idx).ffill(limit=5)
        macro = macro.fillna(0.0)
    except Exception as exc:
        logger.warning("FRED fetch failed (%s) — using zero macro features", exc)
        macro = empty.copy()

    if pit_shift > 0:
        macro = macro.shift(pit_shift)

    return macro.reindex(idx).fillna(0.0)
