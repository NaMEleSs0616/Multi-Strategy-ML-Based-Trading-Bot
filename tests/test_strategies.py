"""Strategy bank unit tests."""

import numpy as np
import pandas as pd

from strategies.base import compound_to_daily, position_returns
from strategies.mean_reversion import mean_reversion_returns
from strategies.stat_arb import select_best_pair, stat_arb_returns
from strategies.vol_breakout import vol_breakout_returns


def _ohlcv(n: int, freq: str = "15min", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + rng.uniform(0.1, 0.5, n),
            "low": close - rng.uniform(0.1, 0.5, n),
            "close": close,
            "volume": rng.integers(1e5, 1e6, n),
        },
        index=idx,
    )


def test_position_returns_pit():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    pos = pd.Series([0, 1, 1, -1, 0], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=idx)
    strat = position_returns(pos, ret, pit_shift_bars=1)
    assert strat.iloc[1] == 0.0
    assert np.isclose(strat.iloc[2], -0.01)


def test_select_best_pair():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rng = np.random.default_rng(3)
    a = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)), index=idx)
    b = a * 0.99 + rng.normal(0, 0.3, 120)
    c = pd.Series(50 + np.cumsum(rng.normal(0, 2, 120)), index=idx)
    prices = {"NVDA": a, "AMD": pd.Series(b, index=idx), "META": c}
    leg_a, leg_b, beta, pval = select_best_pair(prices, ["NVDA", "AMD", "META"])
    assert leg_a in prices and leg_b in prices
    assert beta > 0
    assert 0 <= pval <= 1


def test_stat_arb_runs():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    a = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 120)), index=idx)
    b = a * 0.98 + np.random.default_rng(2).normal(0, 0.5, 120)
    out = stat_arb_returns(a, pd.Series(b, index=idx))
    assert len(out) == 120
    assert out.notna().all()


def test_vol_breakout_runs():
    bars = _ohlcv(200, freq="15min")
    out = vol_breakout_returns(bars)
    assert len(out) == len(bars)


def test_mean_reversion_runs():
    bars = _ohlcv(200, freq="5min")
    out = mean_reversion_returns(bars)
    assert len(out) == len(bars)


def test_compound_to_daily():
    idx = pd.date_range("2024-01-01 09:30", periods=8, freq="15min")
    r = pd.Series(0.001, index=idx)
    daily_idx = pd.date_range("2024-01-01", periods=2, freq="D")
    daily = compound_to_daily(r, daily_idx)
    assert len(daily) == 2
