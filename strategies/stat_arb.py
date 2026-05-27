"""
Cointegration-based pairs trading on daily bars.

Trades the spread between two correlated tech names (e.g. NVDA / AMD).
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from strategies.base import pit_shift, position_returns


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio: y ~ beta * x."""
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)
    if cov[1, 1] < 1e-12:
        return 1.0
    return float(cov[0, 1] / cov[1, 1])


def select_pair(
    prices: dict[str, pd.Series],
    candidates: Tuple[str, str],
) -> Tuple[str, str, float]:
    """Return pair names and hedge ratio (uses cointegration test when enough data)."""
    a, b = candidates
    if a not in prices or b not in prices:
        raise KeyError(f"Missing prices for {candidates}")

    pa = prices[a].astype(float)
    pb = prices[b].astype(float)
    aligned = pd.concat([pa, pb], axis=1, join="inner").dropna()
    if len(aligned) < 60:
        return a, b, estimate_hedge_ratio(pa, pb)

    beta = estimate_hedge_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return a, b, beta


def select_best_pair(
    prices: dict[str, pd.Series],
    universe: Sequence[str],
    *,
    min_obs: int = 60,
) -> Tuple[str, str, float, float]:
    """
    Pick the pair with the lowest Engle–Granger cointegration p-value.

    Returns ``(leg_a, leg_b, hedge_ratio, p_value)``.
    """
    symbols = [s for s in universe if s in prices and len(prices[s].dropna()) >= min_obs]
    if len(symbols) < 2:
        raise ValueError("Need at least two symbols with sufficient price history")

    best: Optional[Tuple[str, str, float, float]] = None
    for a, b in combinations(symbols, 2):
        aligned = pd.concat([prices[a], prices[b]], axis=1, join="inner").dropna()
        if len(aligned) < min_obs:
            continue
        pvalue = float(coint(aligned.iloc[:, 0], aligned.iloc[:, 1])[1])
        beta = estimate_hedge_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])
        if best is None or pvalue < best[3]:
            best = (a, b, beta, pvalue)

    if best is None:
        a, b = symbols[0], symbols[1]
        return a, b, estimate_hedge_ratio(prices[a], prices[b]), 1.0
    return best[0], best[1], best[2], best[3]


def apply_transaction_cost_drag(
    returns: pd.Series,
    position: pd.Series,
    *,
    cost_bps: float,
    legs: int = 2,
) -> pd.Series:
    """Subtract per-leg transaction costs when position changes."""
    if cost_bps <= 0:
        return returns
    turnover = position.diff().abs().fillna(0.0)
    drag = turnover * (cost_bps / 10_000.0) * legs
    return (returns - drag).fillna(0.0)


def stat_arb_returns(
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    hedge_ratio: Optional[float] = None,
    z_window: int = 20,
    entry_z: float = 2.0,
    pit_shift_bars: int = 1,
    transaction_cost_bps: float = 0.0,
) -> pd.Series:
    """
    Mean-revert the cointegration spread; returns are spread P&L (daily).
    """
    aligned = pd.concat([leg_a.rename("a"), leg_b.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < z_window + 5:
        return pd.Series(0.0, index=aligned.index)

    beta = hedge_ratio if hedge_ratio is not None else estimate_hedge_ratio(aligned["a"], aligned["b"])
    spread = np.log(aligned["a"]) - beta * np.log(aligned["b"])
    mu = spread.rolling(z_window, min_periods=z_window).mean()
    sigma = spread.rolling(z_window, min_periods=z_window).std().replace(0, np.nan)
    z = (spread - mu) / sigma

    z_pit = pit_shift(z, pit_shift_bars)
    position = pd.Series(0.0, index=aligned.index)
    position[z_pit > entry_z] = -1.0
    position[z_pit < -entry_z] = 1.0
    position[z_pit.abs() < 0.5] = 0.0

    spread_return = spread.diff().fillna(0.0)
    strat = position_returns(position, spread_return, pit_shift_bars=1).fillna(0.0)
    return apply_transaction_cost_drag(
        strat,
        position,
        cost_bps=transaction_cost_bps,
        legs=2,
    )
