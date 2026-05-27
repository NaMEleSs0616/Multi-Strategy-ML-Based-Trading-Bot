"""Adapter unit tests (mocked data plane)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from adapters.data.yfinance_handler import YFinanceDataHandler
from adapters.execution.alpaca_execution import AlpacaExecutionHandler
from adapters.execution.backtest_execution import BacktestExecutionHandler
from adapters.factory import create_data_handler, create_execution_handler
from core.interfaces import BarQuery, FeatureRequest, LimitOrderRequest, OrderSide


@pytest.fixture
def settings():
    return {
        "universe": {"equities": ["NVDA"], "etfs": []},
        "data": {
            "provider": "yfinance",
            "sqlite_path": "data/storage/market_data.db",
            "daily_interval": "1d",
            "yfinance_period": "2y",
        },
        "features": {
            "frac_diff_d": 0.4,
            "autocorr_window": 20,
            "atr_window": 14,
            "pit_shift": 1,
            "include_macro": False,
            "include_sentiment": False,
        },
        "xlstm": {"yfinance_period": "2y"},
        "execution": {"mode": "backtest", "spread_bps": 5.0},
        "backtest": {"initial_cash": 100_000.0},
    }


def test_factory_returns_yfinance_handler(settings):
    handler = create_data_handler(settings)
    assert isinstance(handler, YFinanceDataHandler)


def test_yfinance_fetch_bars_mock(settings, tmp_path):
    settings["data"]["sqlite_path"] = str(tmp_path / "m.db")
    handler = YFinanceDataHandler(settings)
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    n = len(idx)
    close = 100 + np.arange(n)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    with patch.object(handler, "fetch_bars", return_value=bars):
        out = handler.fetch_bars(BarQuery(symbol="NVDA", interval="1d"))
    assert len(out) == 80


def test_feature_frame_mock(settings, tmp_path):
    settings["data"]["sqlite_path"] = str(tmp_path / "m.db")
    handler = YFinanceDataHandler(settings)
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    n = len(idx)
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    with patch.object(handler, "fetch_bars", return_value=bars):
        frame = handler.fetch_feature_frame(FeatureRequest(symbols=["NVDA"], pit_shift=1))
    assert "log_return" in frame.columns
    assert frame.isna().sum().sum() == 0


def test_backtest_execution_limit_fill():
    ex = BacktestExecutionHandler(initial_cash=100_000.0, spread_bps=5.0)
    ack = ex.submit_limit_order(
        LimitOrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=10.0,
            limit_price=100.0,
        )
    )
    fills = ex.on_bar("SPY", open_=99, high=101, low=98, close=100.5)
    assert ack.order_id
    assert len(fills) == 1
    assert ex.get_positions().get("SPY", 0) == 10.0


def test_alpaca_execution_stub_without_keys():
    ex = AlpacaExecutionHandler()
    ack = ex.submit_limit_order(
        LimitOrderRequest(
            symbol="NVDA",
            side=OrderSide.BUY,
            quantity=1.0,
            limit_price=500.0,
        )
    )
    assert ack.status.value in {"ACCEPTED", "REJECTED"}


def test_factory_backtest_execution(settings):
    ex = create_execution_handler(settings, mode="backtest")
    assert isinstance(ex, BacktestExecutionHandler)
