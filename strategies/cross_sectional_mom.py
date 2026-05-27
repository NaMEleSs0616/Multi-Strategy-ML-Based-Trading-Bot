"""
Cross-sectional momentum: ticker 90-day ROC vs. SPY 90-day ROC.

Allocates long to the ticker whenever its 90-day rate-of-change exceeds
the benchmark's (i.e. the ticker is "winning"). Flat otherwise.

This is a daily, long-only relative-strength leg — it complements the
absolute-trend leg (:mod:`strategies.daily_trend`) by encoding *which*
asset is leading the market rather than whether the market itself is in
uptrend.

Signal
------
    roc_ticker_t = close_ticker_t  / close_ticker_{t-L} - 1
    roc_spy_t    = close_spy_t     / close_spy_{t-L}    - 1
    position_t   = +1 if roc_ticker_{t-1} > roc_spy_{t-1} else 0

Both ROCs are PiT-shifted before signal generation; the realized return
at bar ``t`` is ``position_{t-1} * asset_return_t``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.base import pit_shift, position_returns


def cross_sectional_momentum_returns(
    ticker_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    lookback: int = 90,
    pit_shift_bars: int = 1,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Long ticker when ticker's ``lookback``-day ROC > benchmark's.

    Parameters
    ----------
    ticker_bars
        Daily OHLCV frame for the primary ticker. Must contain ``close``.
    benchmark_bars
        Daily OHLCV frame for the benchmark (typically SPY). Must contain
        ``close``.
    lookback
        ROC window in bars (default 90 daily bars ≈ one quarter).
    pit_shift_bars
        PiT shift applied to both ROCs. Default 1.
    index
        Optional master DatetimeIndex to align the output onto. If omitted,
        the output is aligned to ``ticker_bars.index``.

    Returns
    -------
    pd.Series
        Strategy returns aligned to ``index`` (or ``ticker_bars.index``).
    """
    if "close" not in ticker_bars.columns:
        raise ValueError("ticker_bars must contain a 'close' column")
    if "close" not in benchmark_bars.columns:
        raise ValueError("benchmark_bars must contain a 'close' column")

    out_index = (
        pd.DatetimeIndex(index).normalize()
        if index is not None
        else pd.DatetimeIndex(ticker_bars.index).normalize()
    )

    def _daily_close(bars: pd.DataFrame) -> pd.Series:
        s = bars["close"].astype(float)
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.Series(s.values, index=idx.normalize())

    ticker_close = _daily_close(ticker_bars).reindex(out_index).ffill()
    spy_close = _daily_close(benchmark_bars).reindex(out_index).ffill()

    roc_ticker = ticker_close.pct_change(lookback).fillna(0.0)
    roc_spy = spy_close.pct_change(lookback).fillna(0.0)

    roc_ticker_pit = pit_shift(roc_ticker, pit_shift_bars)
    roc_spy_pit = pit_shift(roc_spy, pit_shift_bars)

    position = pd.Series(0.0, index=out_index)
    position[roc_ticker_pit > roc_spy_pit] = 1.0

    asset_returns = ticker_close.pct_change().fillna(0.0)
    return position_returns(position, asset_returns, pit_shift_bars=1).fillna(0.0)
