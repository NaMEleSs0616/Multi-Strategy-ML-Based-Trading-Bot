"""Unit tests for TradingRoutingEnv per-step reward (no double-counting)."""

from __future__ import annotations

import numpy as np
import pytest

from rl.gym_trading_env import StrategyReturns, TradingRoutingEnv


def _step_reward(
    portfolio_return: float,
    spy_return: float,
    *,
    sortino_weight: float = 0.5,
    outperformance_weight: float = 0.5,
) -> tuple[float, dict]:
    """Run one env step with 100% weight on a single controllable leg."""
    strategies = StrategyReturns(
        stat_arb=np.array([portfolio_return, 0.0]),
        vol_breakout=np.array([0.0, 0.0]),
        mean_reversion=np.array([0.0, 0.0]),
    )
    embeddings = np.zeros((2, 4))
    spy = np.array([spy_return, 0.0])
    env = TradingRoutingEnv(
        embeddings,
        strategies,
        spy,
        sortino_weight=sortino_weight,
        outperformance_weight=outperformance_weight,
        turnover_penalty_lambda=0.0,
        turnover_penalty_bps=0.0,
    )
    env.reset()
    # Softmax → ~[1, 0, 0]: portfolio return equals stat_arb leg at t=0.
    action = np.array([10.0, -10.0, -10.0], dtype=np.float64)
    _, reward, _, _, info = env.step(action)
    return reward, info


def test_beat_spy_day_uses_weighted_blend_not_double_count():
    port, spy = 0.03, 0.01
    reward, info = _step_reward(port, spy)

    expected = 0.5 * port + 0.5 * (port - spy)
    buggy = port + max(0.0, port - spy)  # old formula on green beat-SPY day

    assert reward == pytest.approx(expected)
    assert abs(reward - buggy) > 1e-6
    assert info["sortino_component"] == pytest.approx(0.5 * port)
    assert info["outperformance_component"] == pytest.approx(0.5 * (port - spy))


def test_loss_day_uses_weighted_blend_not_double_count():
    port, spy = -0.02, 0.01
    reward, info = _step_reward(port, spy)

    sortino_step = port * (1.0 + 0.5)  # sortino_weight=0.5
    expected = 0.5 * sortino_step + 0.5 * (port - spy)
    buggy = port - abs(port)  # old formula: 2×portfolio when missing SPY

    assert reward == pytest.approx(expected)
    assert abs(reward - buggy) > 1e-6
    assert info["sortino_step"] == pytest.approx(sortino_step)


def test_green_day_underperforming_spy_penalized_via_alpha():
    port, spy = 0.01, 0.02
    reward, info = _step_reward(port, spy)

    expected = 0.5 * port + 0.5 * (port - spy)  # 0.005 - 0.005 = 0.0
    assert info["alpha"] == pytest.approx(port - spy)
    assert info["alpha"] < 0.0
    assert reward == pytest.approx(expected, abs=1e-9)


def test_outperformance_component_is_signed_not_clipped():
    _, info = _step_reward(0.01, 0.02)
    assert info["outperformance_component"] == pytest.approx(0.5 * (0.01 - 0.02))

    _, info_win = _step_reward(0.03, 0.01)
    assert info_win["outperformance_component"] == pytest.approx(0.5 * 0.02)
