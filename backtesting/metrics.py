"""Risk and performance metrics for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def information_ratio(
    excess_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    r = excess_returns.dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std())
    if std < 1e-12:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def summarize_backtest(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    equity = (1.0 + portfolio_returns.fillna(0.0)).cumprod()
    spy_equity = (1.0 + spy_returns.fillna(0.0)).cumprod()
    excess = portfolio_returns - spy_returns

    return {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "spy_total_return": float(spy_equity.iloc[-1] - 1.0) if len(spy_equity) else 0.0,
        "information_ratio": information_ratio(excess),
        "max_drawdown": max_drawdown(equity),
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "volatility_ann": float(portfolio_returns.std() * np.sqrt(252)) if len(portfolio_returns) > 1 else 0.0,
    }
