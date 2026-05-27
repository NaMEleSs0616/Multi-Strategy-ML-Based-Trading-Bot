"""
Daily momentum (SMA crossover) — long-horizon beta capture.

This leg exists specifically to plug the 60-day intraday blindspot and the
market-neutral bias of the original 3-leg bank. It operates strictly on
daily bars (no 5m/15m dependency), so it survives Alpaca's 60-day intraday
window and is unaffected by the live-WebSocket latency circuit breaker in
``adapters.execution.alpaca_execution``.

Signal
------
Classic 50/200 SMA crossover (a.k.a. "Golden Cross / Death Cross"):

    position_t = +1  if SMA_fast_t > SMA_slow_t   (uptrend)
    position_t = -1  if SMA_fast_t < SMA_slow_t   (downtrend)
    position_t =  0  otherwise

Both SMAs are PiT-shifted before signal generation so the realized return
at bar ``t`` is ``position_{t-1} * asset_return_t``, matching the rest of
the strategy bank.

Returning a ``pd.Series`` aligned to the input ``bars.index`` keeps this
leg behind the same contract as ``stat_arb_returns``, ``vol_breakout_returns``,
and ``mean_reversion_returns`` — so it slots into the bank without any
interface change.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import pit_shift, position_returns


def daily_momentum_returns(
    bars: pd.DataFrame,
    *,
    fast_window: int = 50,
    slow_window: int = 200,
    pit_shift_bars: int = 1,
) -> pd.Series:
    """
    Long-only / short-only SMA crossover on daily closes.

    Parameters
    ----------
    bars
        Daily OHLCV frame with a ``close`` column.
    fast_window, slow_window
        Fast/slow SMA windows (default 50/200). ``slow_window`` must be
        > ``fast_window``.
    pit_shift_bars
        Mandatory PiT shift applied to BOTH SMAs before signal generation.
        Default 1 matches every other strategy leg in the bank.

    Returns
    -------
    pd.Series
        Strategy returns aligned to ``bars.index``. Bars before the slow
        SMA warms up yield 0.0 returns.
    """
    if slow_window <= fast_window:
        raise ValueError(
            f"slow_window ({slow_window}) must exceed fast_window ({fast_window})"
        )
    if "close" not in bars.columns:
        raise ValueError("daily_momentum_returns requires a 'close' column")

    close = bars["close"].astype(float)

    sma_fast = close.rolling(fast_window, min_periods=fast_window).mean()
    sma_slow = close.rolling(slow_window, min_periods=slow_window).mean()

    sma_fast_pit = pit_shift(sma_fast, pit_shift_bars)
    sma_slow_pit = pit_shift(sma_slow, pit_shift_bars)

    position = pd.Series(0.0, index=bars.index)
    position[sma_fast_pit > sma_slow_pit] = 1.0
    position[sma_fast_pit < sma_slow_pit] = -1.0

    asset_returns = close.pct_change().fillna(0.0)
    return position_returns(position, asset_returns, pit_shift_bars=1)
