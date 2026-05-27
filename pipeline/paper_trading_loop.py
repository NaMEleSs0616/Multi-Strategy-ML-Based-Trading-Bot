"""
Single-run paper trading orchestrator (offline / stub execution).

Uses yfinance or SQLite data only; does not call Alpaca WebSocket streams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from adapters.factory import create_data_handler, create_execution_handler
from backtesting.ppo_router import equal_weight_fn, load_weight_fn
from config.settings_store import PROJECT_ROOT, load_settings
from core.interfaces import BarQuery
from core.risk_manager import apply_kelly_from_settings
from pipeline.training_data import build_training_data


@dataclass
class PaperLoopResult:
    timestamp: str
    weights: dict[str, float]
    gross_exposure: float
    equity: float
    symbol: str
    orders_submitted: int
    fills: int
    report_path: Path


def _latest_spy_bars(settings: dict[str, Any]) -> pd.DataFrame:
    handler = create_data_handler(settings)
    bars = handler.fetch_bars(
        BarQuery(symbol="SPY", interval=settings["data"]["daily_interval"])
    )
    if bars.empty:
        from strategies.bank import build_strategy_bank

        bars = build_strategy_bank(settings).primary_bars
    return bars


def run_paper_loop(
    settings: Optional[dict[str, Any]] = None,
    *,
    use_ppo: bool = True,
    equal_weight_fallback: bool = True,
    dry_run: bool = True,
    symbol: str = "SPY",
) -> PaperLoopResult:
    """
    One-shot rebalance: load data/models, Kelly-scale weights, passive limits on ``symbol``.

    ``dry_run=True`` uses :class:`BacktestExecutionHandler` (no Alpaca keys).
    """
    settings = settings or load_settings()
    exec_cfg = settings.setdefault("execution", {})
    if dry_run:
        exec_cfg["mode"] = "backtest"

    data = build_training_data(settings, require_encoder=False)
    matrix = data.strategy_returns.as_matrix()
    n_steps = matrix.shape[0]
    t = n_steps - 1

    n_strategies = int(settings["rl"].get("n_strategies", 3))
    if use_ppo:
        try:
            weight_fn = load_weight_fn(data.embeddings)
        except FileNotFoundError:
            if not equal_weight_fallback:
                raise
            weight_fn = equal_weight_fn(n_strategies)
    else:
        weight_fn = equal_weight_fn(n_strategies)

    weights = weight_fn(t)
    weights = apply_kelly_from_settings(weights, matrix, settings, bar_index=t)
    gross_exposure = float(np.clip(np.sum(np.abs(weights)), 0.0, 1.5))

    bars = _latest_spy_bars(settings)
    if bars.empty:
        raise RuntimeError("No bars available for paper loop execution")

    last = bars.iloc[-1]
    ts = pd.Timestamp(bars.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)

    handler = create_execution_handler(
        settings,
        initial_cash=float(settings.get("backtest", {}).get("initial_cash", 100_000.0)),
    )

    rebalance_threshold = float(
        settings.get("backtest", {}).get("rebalance_threshold", 0.05)
    )
    target_notional = gross_exposure * handler.get_equity()
    close = float(last["close"])
    target_qty = target_notional / max(close, 1e-6)

    positions = handler.get_positions()
    current_qty = positions.get(symbol.upper(), 0.0)
    delta_qty = target_qty - current_qty

    orders_submitted = 0
    fills = 0

    if abs(delta_qty) / max(handler.get_equity() / max(close, 1e-6), 1.0) >= rebalance_threshold:
        side = "BUY" if delta_qty > 0 else "SELL"
        qty = abs(delta_qty)
        if qty >= 1.0:
            handler.submit_passive_from_signal(symbol, side, qty, close)
            orders_submitted = 1

    bar_fills = handler.on_bar(
        symbol,
        open_=float(last.get("open", close)),
        high=float(last.get("high", close)),
        low=float(last.get("low", close)),
        close=close,
        timestamp=ts.to_pydatetime(),
    )
    fills = len(bar_fills)

    report_dir = PROJECT_ROOT / settings.get("backtest", {}).get(
        "report_dir", "backtesting/reports"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "paper_loop_latest.json"

    payload = {
        "timestamp": ts.isoformat(),
        "dry_run": dry_run,
        "symbol": symbol.upper(),
        "weights": {
            "stat_arb": float(weights[0]),
            "vol_breakout": float(weights[1]),
            "mean_reversion": float(weights[2]),
        },
        "gross_exposure": gross_exposure,
        "equity": handler.get_equity(),
        "orders_submitted": orders_submitted,
        "fills": fills,
        "positions": handler.get_positions(),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return PaperLoopResult(
        timestamp=ts.isoformat(),
        weights=payload["weights"],
        gross_exposure=gross_exposure,
        equity=handler.get_equity(),
        symbol=symbol.upper(),
        orders_submitted=orders_submitted,
        fills=fills,
        report_path=report_path,
    )
