"""Portfolio backtest tests."""

import numpy as np
import pandas as pd

from backtesting.portfolio_simulator import simulate_routed_portfolio
from backtesting.ppo_router import equal_weight_fn
from pipeline.training_data import TrainingData
from rl.gym_trading_env import StrategyReturns


def _fake_training_data(n: int = 100) -> TrainingData:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (n, 8))
    return TrainingData(
        embeddings=emb,
        strategy_returns=StrategyReturns(
            stat_arb=rng.normal(0, 0.01, n),
            vol_breakout=rng.normal(0, 0.01, n),
            mean_reversion=rng.normal(0, 0.01, n),
        ),
        spy_returns=rng.normal(0.0002, 0.01, n),
        index=idx,
        features=emb[:, :5],
    )


def test_simulate_routed_portfolio_runs():
    data = _fake_training_data(80)
    result = simulate_routed_portfolio(data, equal_weight_fn(3))
    assert len(result.equity) == 80
    assert "information_ratio" in result.metrics
    assert result.weights.shape == (80, 3)


def test_weights_sum_to_one():
    data = _fake_training_data(20)
    result = simulate_routed_portfolio(data, equal_weight_fn(3))
    sums = result.weights.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
