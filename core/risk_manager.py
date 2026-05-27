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

    # --- Momentum-aware sizing (anti "Lazy Agent" hot-fix) ----------------
    # When a leg's rolling win-rate clears ``momentum_win_rate_threshold``
    # the fractional Kelly is multiplied by ``momentum_boost`` and floored
    # at ``min_momentum_allocation`` so the PPO agent retains real sizing
    # in clear trending regimes instead of being throttled to cash.
    min_momentum_allocation: float = 0.0
    momentum_win_rate_threshold: float = 0.55
    momentum_boost: float = 1.5


class RiskManager:
    """
    Rolling Kelly cap per strategy leg.

    For each leg over a ``window``-day history:
      W = win rate, R = avg win / avg loss
      K = W - (1 - W) / R
      fractional_kelly = ``fraction`` * K  (default 0.5 * K)

    A momentum-regime override scales ``fractional_kelly`` up (and floors
    it at ``min_momentum_allocation``) when the leg's recent win-rate
    exceeds ``momentum_win_rate_threshold``. This prevents the "Lazy
    Agent" pathology where Kelly throttles the PPO router to cash in
    obvious trending regimes.

    ``apply_kelly_cap`` scales softmax weights down when Kelly is low.
    """

    def __init__(self, config: Optional[KellyConfig] = None) -> None:
        self.config = config or KellyConfig()

    @staticmethod
    def _leg_win_rate(returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        nonzero = r[r != 0.0]
        if nonzero.size == 0:
            return 0.0
        return float((nonzero > 0).mean())

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

    def _momentum_adjusted_fk(self, fk: float, segment: np.ndarray) -> float:
        """Boost fractional Kelly and apply momentum floor when win-rate is high."""
        cfg = self.config
        if cfg.min_momentum_allocation <= 0.0 and cfg.momentum_boost <= 1.0:
            return fk
        win_rate = self._leg_win_rate(segment)
        if win_rate >= cfg.momentum_win_rate_threshold:
            fk = max(fk * cfg.momentum_boost, cfg.min_momentum_allocation)
        return fk

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
            fk = self._momentum_adjusted_fk(fk, segment)
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

    @staticmethod
    def cap_max_weight(w: np.ndarray, *, max_weight: float, eps: float = 1e-12) -> np.ndarray:
        """
        Enforce a hard diversification cap: no single strategy may exceed ``max_weight``.

        The remainder is redistributed proportionally across uncapped weights when possible,
        otherwise spread evenly across the other legs. Always returns a vector summing to 1.
        """
        w = np.asarray(w, dtype=np.float64).reshape(-1).copy()
        n = w.size
        if n == 0:
            return w
        cap = float(max_weight)
        if cap <= 0.0:
            return np.ones(n, dtype=np.float64) / n
        if cap >= 1.0:
            total = float(w.sum())
            return w / total if total > eps else (np.ones(n, dtype=np.float64) / n)

        # Iteratively cap until stable (handles multiple > cap).
        uncapped = np.ones(n, dtype=bool)
        w = np.maximum(w, 0.0)
        while True:
            over = (w > cap) & uncapped
            if not bool(np.any(over)):
                break
            w[over] = cap
            uncapped[over] = False

            remaining = 1.0 - float(w[~uncapped].sum())
            if remaining <= 0.0:
                # Too many capped weights; renormalize capped mass.
                total = float(w.sum())
                return w / total if total > eps else (np.ones(n, dtype=np.float64) / n)

            base = w[uncapped]
            base_sum = float(base.sum())
            if base_sum > eps:
                w[uncapped] = base / base_sum * remaining
            else:
                # No positive uncapped weights: distribute remainder equally.
                k = int(np.sum(uncapped))
                if k == 0:
                    total = float(w.sum())
                    return w / total if total > eps else (np.ones(n, dtype=np.float64) / n)
                w[uncapped] = remaining / k

        total = float(w.sum())
        return w / total if total > eps else (np.ones(n, dtype=np.float64) / n)


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
            min_momentum_allocation=float(
                risk_cfg.get("min_momentum_allocation", 0.0)
            ),
            momentum_win_rate_threshold=float(
                risk_cfg.get("momentum_win_rate_threshold", 0.55)
            ),
            momentum_boost=float(risk_cfg.get("momentum_boost", 1.5)),
        )
    )
    w = rm.apply_kelly_cap(weights, strategy_matrix, end_index=bar_index)

    # Live/paper-only diversification floor: prevent 100% cornering into a single leg.
    exec_cfg = settings.get("execution", {}) or {}
    if str(exec_cfg.get("mode", "backtest")).lower() == "alpaca" and bool(exec_cfg.get("paper", True)):
        cap = float(risk_cfg.get("max_single_weight_live", 0.60))
        w = rm.cap_max_weight(w, max_weight=cap, eps=rm.config.eps)

    return w
