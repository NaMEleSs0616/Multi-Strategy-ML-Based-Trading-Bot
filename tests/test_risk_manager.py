"""RiskManager fractional Kelly tests."""

import numpy as np

from core.risk_manager import KellyConfig, RiskManager
from rl.gym_trading_env import StrategyReturns


def test_kelly_reduces_weights_on_poor_leg():
    rng = np.random.default_rng(0)
    n = 60
    good = rng.normal(0.002, 0.01, n)
    bad = rng.normal(-0.003, 0.01, n)
    matrix = np.stack([good, bad, rng.normal(0, 0.01, n)], axis=1)

    rm = RiskManager(KellyConfig(window=30, fraction=0.5))
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    capped = rm.apply_kelly_cap(w, matrix, end_index=n - 1)
    assert abs(capped.sum() - 1.0) < 1e-6
    assert capped[1] < capped[0]


def test_strategy_returns_matrix_input():
    n = 40
    sr = StrategyReturns(
        stat_arb=np.full(n, 0.01),
        vol_breakout=np.full(n, -0.01),
        mean_reversion=np.zeros(n),
    )
    rm = RiskManager(KellyConfig(window=20, fraction=0.5))
    mult = rm.kelly_multipliers(sr, end_index=n - 1)
    assert mult.shape == (3,)
    assert np.all(mult >= 0)
