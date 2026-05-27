"""Historical OHLCV via Alpaca Market Data API."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PERIOD_RE = re.compile(r"^(\d+)(y|mo|wk|d)$", re.I)


def period_to_timedelta(period: str) -> timedelta:
    """Convert yfinance-style period strings (``2y``, ``60d``) to ``timedelta``."""
    match = _PERIOD_RE.match(period.strip())
    if not match:
        return timedelta(days=730)
    amount, unit = int(match.group(1)), match.group(2).lower()
    if unit == "y":
        return timedelta(days=amount * 365)
    if unit == "mo":
        return timedelta(days=amount * 30)
    if unit == "wk":
        return timedelta(days=amount * 7)
    return timedelta(days=amount)


def resolve_alpaca_range(
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    period: Optional[str] = None,
    default_period: str = "2y",
) -> tuple[datetime, datetime]:
    """Resolve Alpaca ``start``/``end`` from explicit bounds or a lookback period."""
    end_dt = end or datetime.now(timezone.utc).replace(tzinfo=None)
    if start is not None:
        return start, end_dt
    lookback = period or default_period
    return end_dt - period_to_timedelta(lookback), end_dt


def fetch_alpaca_bars(
    symbol: str,
    interval: str = "1d",
    *,
    period: Optional[str] = "2y",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca; returns empty frame when keys or data are missing."""
    api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        logger.warning("alpaca-py not installed")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    tf_map = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
    }
    timeframe = tf_map.get(interval)
    if timeframe is None:
        logger.warning("Unsupported Alpaca interval %s", interval)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    start_dt, end_dt = resolve_alpaca_range(
        start=start,
        end=end,
        period=period,
        default_period="2y" if interval in {"1d", "1wk"} else "60d",
    )

    client = StockHistoricalDataClient(api_key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    bars = bars.rename(columns=str.lower)
    out = bars[["open", "high", "low", "close", "volume"]].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()
