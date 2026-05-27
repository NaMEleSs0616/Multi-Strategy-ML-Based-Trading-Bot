"""
Daily trend follower (long-only SMA crossover) — pure macro beta capture.

Differs from :mod:`strategies.daily_momentum` (which is long/short/flat) by
emitting **only long or flat** positions. This is the textbook "Golden /
Death Cross" trend filter: it expresses uptrends with full conviction and
sits in cash during downtrends, which gives the PPO router a clean,
asymmetric long-side beta leg.

Signal
------
    position_t = +1  if  SMA_fast_t > SMA_slow_t   (uptrend)
    position_t =  0  otherwise

Both SMAs are PiT-shifted before signal generation, mirroring every other
leg in the bank.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import pit_shift, position_returns


def daily_trend_returns(
    bars: pd.DataFrame,
    *,
    fast_window: int = 50,
    slow_window: int = 200,
    pit_shift_bars: int = 1,
) -> pd.Series:
    """
    Long-only 50/200 SMA crossover on daily closes.

    Parameters
    ----------
    bars
        Daily OHLCV frame with a ``close`` column.
    fast_window, slow_window
        Fast/slow SMA windows (default 50/200). ``slow_window`` must be
        strictly greater than ``fast_window``.
    pit_shift_bars
        Mandatory PiT shift on both SMAs. Default 1 keeps the contract
        identical to other strategy legs.

    Returns
    -------
    pd.Series
        Strategy returns aligned to ``bars.index``. Pre-warmup bars produce
        0.0 returns (signal is implicitly flat).
    """
    if slow_window <= fast_window:
        raise ValueError(
            f"slow_window ({slow_window}) must exceed fast_window ({fast_window})"
        )
    if "close" not in bars.columns:
        raise ValueError("daily_trend_returns requires a 'close' column")

    close = bars["close"].astype(float)
    sma_fast = close.rolling(fast_window, min_periods=fast_window).mean()
    sma_slow = close.rolling(slow_window, min_periods=slow_window).mean()

    sma_fast_pit = pit_shift(sma_fast, pit_shift_bars)
    sma_slow_pit = pit_shift(sma_slow, pit_shift_bars)

    position = pd.Series(0.0, index=bars.index)
    position[sma_fast_pit > sma_slow_pit] = 1.0  # uptrend: long, else flat

    asset_returns = close.pct_change().fillna(0.0)
    return position_returns(position, asset_returns, pit_shift_bars=1)
