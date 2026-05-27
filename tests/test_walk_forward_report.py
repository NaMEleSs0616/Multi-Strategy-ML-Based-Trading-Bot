"""Walk-forward OOS report tests."""

import numpy as np
import pandas as pd

from data.harvester.storage import BarStore
from pipeline.walk_forward_report import run_walk_forward_report
from tests.conftest import seed_market_db


def test_walk_forward_report_equal_weight(tmp_path):
    db = tmp_path / "market.db"
    seed_market_db(db)
    settings = {
        "universe": {"equities": ["NVDA", "AMD"], "etfs": ["SPY"]},
        "data": {
            "sqlite_path": str(db),
            "daily_interval": "1d",
            "intraday_intervals": ["5m", "15m"],
            "provider": "yfinance",
        },
        "features": {
            "frac_diff_d": 0.4,
            "autocorr_window": 20,
            "atr_window": 14,
            "pit_shift": 1,
        },
        "strategies": {"pair": ["NVDA", "AMD"], "transaction_cost_bps": 0.0},
        "rl": {
            "embedding_dim": 8,
            "n_strategies": 3,
            "turnover_penalty_lambda": 0.1,
            "purge_embargo_bars": 2,
        },
        "walk_forward": {
            "n_splits": 2,
            "min_train_size": 80,
            "test_size": 40,
        },
        "risk": {"enabled": False},
        "xlstm": {"checkpoint_dir": "models/xlstm/checkpoints"},
        "backtest": {"report_dir": str(tmp_path / "reports")},
    }
    result = run_walk_forward_report(
        settings,
        use_ppo=False,
        export_csv=True,
    )
    assert result.report_path.exists()
    assert result.aggregate["n_folds"] >= 1
    assert "mean_information_ratio" in result.aggregate
    if result.csv_dir:
        assert any(result.csv_dir.glob("fold_*.csv"))
