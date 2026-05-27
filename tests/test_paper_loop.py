"""Paper trading loop dry-run tests."""

from pipeline.paper_trading_loop import run_paper_loop
from tests.conftest import seed_market_db


def test_paper_loop_dry_run(tmp_path):
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
        "strategies": {"pair": ["NVDA", "AMD"]},
        "rl": {"embedding_dim": 8, "n_strategies": 3, "turnover_penalty_lambda": 0.1},
        "xlstm": {"checkpoint_dir": "models/xlstm/checkpoints"},
        "risk": {"enabled": False},
        "execution": {"mode": "backtest", "spread_bps": 5.0},
        "backtest": {
            "initial_cash": 50_000.0,
            "rebalance_threshold": 0.01,
            "report_dir": str(tmp_path / "reports"),
        },
    }
    result = run_paper_loop(settings, use_ppo=False, dry_run=True)
    assert result.report_path.exists()
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5
    assert result.equity > 0
