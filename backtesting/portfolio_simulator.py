"""
Bar-by-bar routed portfolio simulation (non-vectorized).

Matches the RL reward: excess return vs SPY minus turnover penalty on weight changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from adapters.factory import create_execution_handler
from backtesting.metrics import summarize_backtest
from core.interfaces import AbstractExecutionHandler
from pipeline.training_data import TrainingData
from rl.gym_trading_env import StrategyReturns


@dataclass
class PortfolioBacktestResult:
    index: pd.DatetimeIndex
    portfolio_returns: pd.Series
    spy_returns: pd.Series
    excess_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    equity: pd.Series
    metrics: dict[str, float]


def simulate_routed_portfolio(
    data: TrainingData,
    weight_fn: Callable[[int], np.ndarray],
    *,
    turnover_penalty_lambda: float = 0.0,
    apply_penalty_to_returns: bool = False,
    settings: Optional[dict] = None,
) -> PortfolioBacktestResult:
    """
    Event-style loop over bars (no pandas vectorized backtest).

    Parameters
    ----------
    apply_penalty_to_returns
        If True, subtract λ·turnover from realized returns (RL training reward).
        If False, report gross returns and turnover separately (evaluation).
    """
    matrix = data.strategy_returns.as_matrix()
    n_steps, n_strategies = matrix.shape
    spy = data.spy_returns.reshape(-1)
    index = pd.DatetimeIndex(data.index)

    port_rets = np.zeros(n_steps, dtype=np.float64)
    turnovers = np.zeros(n_steps, dtype=np.float64)
    weights_hist = np.zeros((n_steps, n_strategies), dtype=np.float64)

    w_prev = weight_fn(0)
    weights_hist[0] = w_prev

    for t in range(n_steps):
        w_t = weight_fn(t)
        if settings is not None:
            from core.risk_manager import apply_kelly_from_settings

            w_t = apply_kelly_from_settings(w_t, matrix, settings, bar_index=t)
        weights_hist[t] = w_t
        turnover = float(np.sum(np.abs(w_t - w_prev)))
        turnovers[t] = turnover

        gross = float(np.dot(w_t, matrix[t]))
        if apply_penalty_to_returns:
            port_rets[t] = gross - turnover_penalty_lambda * turnover
        else:
            port_rets[t] = gross

        w_prev = w_t

    portfolio_returns = pd.Series(port_rets, index=index, name="portfolio")
    spy_returns = pd.Series(spy, index=index, name="spy")
    excess_returns = portfolio_returns - spy_returns
    turnover_s = pd.Series(turnovers, index=index, name="turnover")
    weights = pd.DataFrame(
        weights_hist,
        index=index,
        columns=["stat_arb", "vol_breakout", "mean_reversion"],
    )
    equity = (1.0 + portfolio_returns).cumprod()
    metrics = summarize_backtest(portfolio_returns, spy_returns, turnover_s)

    return PortfolioBacktestResult(
        index=index,
        portfolio_returns=portfolio_returns,
        spy_returns=spy_returns,
        excess_returns=excess_returns,
        turnover=turnover_s,
        weights=weights,
        equity=equity,
        metrics=metrics,
    )


def simulate_routed_portfolio_with_execution(
    data: TrainingData,
    weight_fn: Callable[[int], np.ndarray],
    bars: pd.DataFrame,
    *,
    symbol: str = "SPY",
    settings: Optional[dict] = None,
    execution_handler: Optional[AbstractExecutionHandler] = None,
    rebalance_threshold: float = 0.05,
) -> tuple[PortfolioBacktestResult, AbstractExecutionHandler]:
    """
    Portfolio simulation with optional rebalance via :class:`BacktestExecutionHandler`.

    Returns portfolio metrics (return-based) plus the execution handler state.
    """
    settings = settings or {}
    portfolio = simulate_routed_portfolio(
        data,
        weight_fn,
        settings=settings,
    )

    handler = execution_handler or create_execution_handler(settings)
    last_exposure = 0.0
    sym = symbol.upper()
    equity_rows: list[tuple[pd.Timestamp, float]] = []

    for t, (_, row) in enumerate(bars.iterrows()):
        w_t = weight_fn(t)
        if settings:
            from core.risk_manager import apply_kelly_from_settings

            matrix = data.strategy_returns.as_matrix()
            w_t = apply_kelly_from_settings(w_t, matrix, settings, bar_index=t)

        exposure = float(np.clip(np.sum(np.abs(w_t)), 0.0, 1.5))
        if abs(exposure - last_exposure) >= rebalance_threshold:
            close = float(row["close"])
            equity = handler.get_equity()
            target_qty = exposure * equity / max(close, 1e-6)
            current = handler.get_positions().get(sym, 0.0)
            delta = target_qty - current
            if abs(delta) >= 1.0:
                side = "BUY" if delta > 0 else "SELL"
                handler.submit_passive_from_signal(sym, side, abs(delta), close)
            last_exposure = exposure

        ts = pd.Timestamp(row.name)
        if ts.tzinfo is None:
            from datetime import timezone

            ts = ts.tz_localize(timezone.utc)
        handler.on_bar(
            sym,
            open_=float(row.get("open", row["close"])),
            high=float(row.get("high", row["close"])),
            low=float(row.get("low", row["close"])),
            close=float(row["close"]),
            timestamp=ts.to_pydatetime(),
        )
        equity_rows.append((pd.Timestamp(row.name), handler.get_equity()))

    handler._equity_curve = pd.Series(  # type: ignore[attr-defined]
        [e for _, e in equity_rows],
        index=pd.DatetimeIndex([ts for ts, _ in equity_rows]),
        name="equity",
    )

    return portfolio, handler
