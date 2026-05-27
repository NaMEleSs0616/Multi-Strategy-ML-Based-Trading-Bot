"""
Mean reversion on intraday bars (default 5m): RSI + Bollinger Band extremes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import pit_shift, position_returns


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def mean_reversion_returns(
    bars: pd.DataFrame,
    *,
    rsi_window: int = 14,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
    bb_window: int = 20,
    bb_std: float = 2.0,
    pit_shift_bars: int = 1,
) -> pd.Series:
    close = bars["close"].astype(float)
    rsi = _rsi(close, rsi_window)
    rsi_pit = pit_shift(rsi, pit_shift_bars)

    mid = close.rolling(bb_window, min_periods=bb_window).mean()
    std = close.rolling(bb_window, min_periods=bb_window).std().replace(0, np.nan)
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    close_pit = pit_shift(close, pit_shift_bars)
    upper_pit = pit_shift(upper, pit_shift_bars)
    lower_pit = pit_shift(lower, pit_shift_bars)

    position = pd.Series(0.0, index=bars.index)
    long_sig = (rsi_pit < rsi_low) | (close_pit < lower_pit)
    short_sig = (rsi_pit > rsi_high) | (close_pit > upper_pit)
    position[long_sig] = 1.0
    position[short_sig] = -1.0

    asset_returns = close.pct_change().fillna(0.0)
    return position_returns(position, asset_returns, pit_shift_bars=1)
