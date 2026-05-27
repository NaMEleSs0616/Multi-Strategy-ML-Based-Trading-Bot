"""
Fractional Kelly overlay on PPO softmax weights (per strategy leg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rl.gym_trading_env import StrategyReturns


@dataclass
class KellyConfig:
    window: int = 30
    fraction: float = 0.5
    min_kelly: float = 0.0
    max_kelly: float = 1.0
    eps: float = 1e-8


class RiskManager:
    """
    Rolling Kelly cap per strategy leg.

    For each leg over a ``window``-day history:
      W = win rate, R = avg win / avg loss
      K = W - (1 - W) / R
      fractional_kelly = ``fraction`` * K  (default 0.5 * K)

    ``apply_kelly_cap`` scales softmax weights down when Kelly is low.
    """

    def __init__(self, config: Optional[KellyConfig] = None) -> None:
        self.config = config or KellyConfig()

    @staticmethod
    def _leg_kelly(returns: np.ndarray, *, eps: float) -> float:
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        if r.size < 2:
            return 0.0
        wins = r[r > 0]
        losses = r[r < 0]
        if wins.size == 0 or losses.size == 0:
            w = float((r > 0).mean())
            return max(0.0, w - 0.5)
        w = float(wins.size / r.size)
        avg_win = float(wins.mean())
        avg_loss = float(np.abs(losses.mean()))
        r_ratio = avg_win / max(avg_loss, eps)
        k = w - (1.0 - w) / max(r_ratio, eps)
        return float(k)

    def kelly_multipliers(
        self,
        strategy_returns_history: np.ndarray,
        *,
        end_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        strategy_returns_history
            Shape ``(T, n_strategies)`` or :class:`StrategyReturns` fields.
        end_index
            Use returns up to this bar (exclusive end slice for rolling window).
        """
        if isinstance(strategy_returns_history, StrategyReturns):
            matrix = strategy_returns_history.as_matrix()
        else:
            matrix = np.asarray(strategy_returns_history, dtype=np.float64)

        t_end = matrix.shape[0] if end_index is None else int(end_index) + 1
        window = self.config.window
        n_strategies = matrix.shape[1]
        multipliers = np.ones(n_strategies, dtype=np.float64)

        for j in range(n_strategies):
            start = max(0, t_end - window)
            segment = matrix[start:t_end, j]
            k = self._leg_kelly(segment, eps=self.config.eps)
            fk = self.config.fraction * k
            fk = float(np.clip(fk, self.config.min_kelly, self.config.max_kelly))
            multipliers[j] = max(0.0, min(1.0, fk))

        return multipliers

    def apply_kelly_cap(
        self,
        ppo_weights: np.ndarray,
        strategy_returns_history: np.ndarray,
        *,
        end_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Scale/cap softmax weights using per-leg fractional Kelly.

        Weights are renormalized to sum to 1 when positive mass remains.
        """
        w = np.asarray(ppo_weights, dtype=np.float64).reshape(-1).copy()
        mult = self.kelly_multipliers(strategy_returns_history, end_index=end_index)
        w = w * mult
        total = w.sum()
        if total > self.config.eps:
            w = w / total
        else:
            n = w.size
            w = np.ones(n, dtype=np.float64) / n
        return w


def apply_kelly_from_settings(
    weights: np.ndarray,
    strategy_matrix: np.ndarray,
    settings: dict,
    *,
    bar_index: int,
) -> np.ndarray:
    """Helper used by env / backtest when ``risk.enabled`` in settings."""
    risk_cfg = settings.get("risk", {})
    if not risk_cfg.get("enabled", True):
        return weights
    rm = RiskManager(
        KellyConfig(
            window=int(risk_cfg.get("kelly_window", 30)),
            fraction=float(risk_cfg.get("kelly_fraction", 0.5)),
        )
    )
    return rm.apply_kelly_cap(weights, strategy_matrix, end_index=bar_index)
