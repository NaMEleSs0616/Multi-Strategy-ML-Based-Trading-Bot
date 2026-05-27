"""Historical OHLCV via yfinance."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def fetch_yfinance_bars(
    symbol: str,
    interval: str = "1d",
    *,
    period: Optional[str] = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    import yfinance as yf

    kwargs: dict = {"interval": interval, "progress": False, "auto_adjust": True}
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period or "2y"

    raw = yf.download(symbol, **kwargs)
    if raw.empty:
        return raw

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    out = raw[keep].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()
