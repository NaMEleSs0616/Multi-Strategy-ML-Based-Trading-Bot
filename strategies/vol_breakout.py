"""
Volatility breakout: ATR momentum on intraday bars (default 15m).
"""

from __future__ import annotations

import pandas as pd

from strategies.base import pit_shift, position_returns


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def vol_breakout_returns(
    bars: pd.DataFrame,
    *,
    atr_window: int = 14,
    atr_mult: float = 1.5,
    lookback: int = 20,
    pit_shift_bars: int = 1,
) -> pd.Series:
    """
    Long when close breaks above prior range high + k*ATR; short on breakdown.
    """
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    tr = _true_range(high, low, close)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    atr_pit = pit_shift(atr, pit_shift_bars)

    range_high = high.rolling(lookback, min_periods=lookback).max()
    range_low = low.rolling(lookback, min_periods=lookback).min()
    range_high_pit = pit_shift(range_high, pit_shift_bars)
    range_low_pit = pit_shift(range_low, pit_shift_bars)

    prev_close = pit_shift(close, pit_shift_bars)
    upper = range_high_pit + atr_mult * atr_pit
    lower = range_low_pit - atr_mult * atr_pit

    position = pd.Series(0.0, index=bars.index)
    position[prev_close > upper] = 1.0
    position[prev_close < lower] = -1.0

    asset_returns = close.pct_change().fillna(0.0)
    return position_returns(position, asset_returns, pit_shift_bars=1)
