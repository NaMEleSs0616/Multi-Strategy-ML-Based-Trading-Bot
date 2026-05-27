"""
Factory for data and execution handlers (dependency injection entry point).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from core.interfaces import AbstractDataHandler, AbstractExecutionHandler
from adapters.data.alpaca_data_handler import AlpacaDataHandler
from adapters.data.yfinance_handler import YFinanceDataHandler
from adapters.execution.alpaca_execution import AlpacaExecutionHandler
from adapters.execution.backtest_execution import BacktestExecutionHandler


Provider = Literal["yfinance", "alpaca"]
ExecutionMode = Literal["backtest", "alpaca", "stub"]


def create_data_handler(
    settings: dict[str, Any],
    *,
    provider: Optional[Provider] = None,
) -> AbstractDataHandler:
    """Return the configured data handler (default: yfinance)."""
    provider = provider or settings.get("data", {}).get("provider", "yfinance")
    if provider == "alpaca":
        handler = AlpacaDataHandler(settings)
        if not getattr(handler, "_available", False):
            return YFinanceDataHandler(settings)
        return handler
    return YFinanceDataHandler(settings)


def create_execution_handler(
    settings: dict[str, Any],
    *,
    mode: Optional[ExecutionMode] = None,
    initial_cash: Optional[float] = None,
) -> AbstractExecutionHandler:
    """Return execution handler for backtest or live Alpaca."""
    mode = mode or settings.get("execution", {}).get("mode", "backtest")
    exec_cfg = settings.get("execution", {})

    if mode == "alpaca":
        return AlpacaExecutionHandler(
            paper=bool(exec_cfg.get("paper", True)),
            settings=settings,
            websockets_connected=bool(exec_cfg.get("websockets_connected", False)),
        )

    cash = initial_cash or float(settings.get("backtest", {}).get("initial_cash", 100_000.0))
    return BacktestExecutionHandler(
        initial_cash=cash,
        spread_bps=float(exec_cfg.get("spread_bps", 5.0)),
    )
