"""
Run routed portfolio backtest and optional event-driven SPY execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtesting.portfolio_simulator import (
    PortfolioBacktestResult,
    simulate_routed_portfolio,
    simulate_routed_portfolio_with_execution,
)
from backtesting.ppo_router import equal_weight_fn, load_weight_fn
from config.settings_store import PROJECT_ROOT, load_settings
from pipeline.training_data import build_training_data


@dataclass
class FullBacktestResult:
    portfolio: PortfolioBacktestResult
    execution_equity: Optional[pd.DataFrame]
    report_path: Optional[Path]


def run_portfolio_backtest(
    settings: Optional[dict[str, Any]] = None,
    *,
    use_ppo: bool = True,
    equal_weight_fallback: bool = True,
    run_execution: bool = False,
) -> FullBacktestResult:
    settings = settings or load_settings()
    rl_cfg = settings["rl"]

    data = build_training_data(settings, require_encoder=False)

    if use_ppo:
        try:
            weight_fn = load_weight_fn(data.embeddings)
        except FileNotFoundError:
            if not equal_weight_fallback:
                raise
            weight_fn = equal_weight_fn(int(rl_cfg.get("n_strategies", 3)))
    else:
        weight_fn = equal_weight_fn(int(rl_cfg.get("n_strategies", 3)))

    portfolio = simulate_routed_portfolio(
        data,
        weight_fn,
        turnover_penalty_lambda=float(rl_cfg.get("turnover_penalty_lambda", 0.1)),
        apply_penalty_to_returns=False,
        settings=settings,
    )

    execution_equity = None
    if run_execution:
        from adapters.factory import create_data_handler
        from core.interfaces import BarQuery

        data_handler = create_data_handler(settings)
        spy_bars = data_handler.fetch_bars(
            BarQuery(symbol="SPY", interval=settings["data"]["daily_interval"])
        )
        if spy_bars.empty:
            from strategies.bank import build_strategy_bank

            spy_bars = build_strategy_bank(settings).primary_bars

        aligned = spy_bars.reindex(pd.DatetimeIndex(data.index)).dropna(subset=["close"])
        if len(aligned) < 30:
            raise ValueError("Insufficient SPY bars aligned to backtest index for execution simulation")

        _, exec_handler = simulate_routed_portfolio_with_execution(
            data,
            weight_fn,
            aligned,
            settings=settings,
            rebalance_threshold=float(
                settings.get("backtest", {}).get("rebalance_threshold", 0.05)
            ),
        )
        execution_equity = pd.DataFrame(
            {"equity": getattr(exec_handler, "_equity_curve", pd.Series(dtype=float))}
        )

    report_dir = PROJECT_ROOT / settings.get("backtest", {}).get("report_dir", "backtesting/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "latest_backtest.json"
    payload = {
        "metrics": portfolio.metrics,
        "n_bars": len(portfolio.index),
        "policy": "ppo" if use_ppo else "equal_weight",
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return FullBacktestResult(
        portfolio=portfolio,
        execution_equity=execution_equity,
        report_path=report_path,
    )


if __name__ == "__main__":
    result = run_portfolio_backtest(run_execution=False)
    m = result.portfolio.metrics
    print(f"IR={m['information_ratio']:.3f} total_return={m['total_return']:.2%} max_dd={m['max_drawdown']:.2%}")
    print(f"Report: {result.report_path}")
