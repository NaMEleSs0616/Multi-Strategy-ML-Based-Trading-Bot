"""End-to-end strategy bank with synthetic SQLite data."""

import numpy as np
import pandas as pd

from strategies.bank import build_strategy_bank
from tests.conftest import seed_market_db


def test_build_strategy_bank(tmp_path):
    db = tmp_path / "market.db"
    seed_market_db(db)
    settings = {
        "universe": {"equities": ["NVDA", "AMD"], "etfs": ["SPY"]},
        "data": {
            "sqlite_path": str(db),
            "daily_interval": "1d",
            "intraday_intervals": ["5m", "15m"],
        },
        "features": {
            "frac_diff_d": 0.4,
            "autocorr_window": 20,
            "atr_window": 14,
            "pit_shift": 1,
        },
        "strategies": {"pair": ["NVDA", "AMD"]},
    }
    bundle = build_strategy_bank(settings, primary_symbol="NVDA")
    n = len(bundle.master_index)
    assert bundle.strategy_returns.stat_arb.shape[0] == n
    assert bundle.spy_returns.shape[0] == n
