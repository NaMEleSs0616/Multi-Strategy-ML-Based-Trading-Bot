"""
Custom Gymnasium environment for PPO strategy routing.

Observation: frozen xLSTM embedding (precomputed or supplied via callback).
Action: continuous logits → softmax capital weights across the strategy bank.
Reward: excess return vs SPY minus L1 turnover penalty (cost-aware).

Walk-forward evaluation uses :class:`PurgedWalkForwardSplitter` with embargo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.validation.purge import PurgedWalkForwardSplitter, WalkForwardFold, purge_indices


@dataclass
class StrategyReturns:
    """Per-strategy simple returns aligned to the master timeline."""

    stat_arb: np.ndarray
    vol_breakout: np.ndarray
    mean_reversion: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.stack(
            [self.stat_arb, self.vol_breakout, self.mean_reversion],
            axis=1,
        )


class TradingRoutingEnv(gym.Env):
    """
    RL routing environment over a fixed strategy bank.

    Parameters
    ----------
    embeddings
        (T, D) frozen xLSTM state vectors, strictly PiT-aligned per row.
    strategy_returns
        StrategyReturns with shape (T,) per leg.
    spy_returns
        (T,) SPY simple returns aligned with embeddings.
    walk_forward_fold
        Optional fold; when set, episodes are confined to purged test indices.
    turnover_penalty_lambda
        λ in R_t = (R_p - R_SPY) - λ ||w_t - w_{t-1}||_1
    purge_embargo
        Embargo bars when constructing custom train/test masks via
        :meth:`set_walk_forward_fold`.
    episode_length
        Max steps per episode before auto-termination (None = full fold).
    seed
        RNG seed for Gymnasium.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        embeddings: np.ndarray,
        strategy_returns: StrategyReturns,
        spy_returns: np.ndarray,
        *,
        walk_forward_fold: Optional[WalkForwardFold] = None,
        turnover_penalty_lambda: float = 0.1,
        purge_embargo: int = 5,
        episode_length: Optional[int] = None,
        seed: Optional[int] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self._embeddings = np.asarray(embeddings, dtype=np.float64)
        self._strategy_matrix = strategy_returns.as_matrix().astype(np.float64)
        self._spy = np.asarray(spy_returns, dtype=np.float64).reshape(-1)

        n_steps, n_strategies = self._strategy_matrix.shape
        if self._embeddings.shape[0] != n_steps:
            raise ValueError("embeddings length must match strategy returns")
        if self._spy.shape[0] != n_steps:
            raise ValueError("spy_returns length must match strategy returns")
        if n_strategies < 2:
            raise ValueError("strategy bank must contain at least two legs")

        self._embedding_dim = int(self._embeddings.shape[1])
        self._n_strategies = n_strategies
        self.turnover_penalty_lambda = turnover_penalty_lambda
        self.purge_embargo = purge_embargo
        self.episode_length = episode_length

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._embedding_dim,),
            dtype=np.float64,
        )
        # Logits before softmax; unconstrained Box is standard for SB3 continuous control.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_strategies,),
            dtype=np.float64,
        )

        self._fold = walk_forward_fold
        self._active_indices = self._resolve_indices(self._fold)
        self._cursor = 0
        self._step_in_episode = 0
        self._weights = np.ones(self._n_strategies, dtype=np.float64) / self._n_strategies
        self._portfolio_returns: list[float] = []
        self._excess_returns: list[float] = []

        self._cumulative_excess = 0.0
        self._excess_sq_sum = 0.0
        self._info_ratio_steps = 0
        self._settings = settings

    @staticmethod
    def _resolve_indices(fold: Optional[WalkForwardFold]) -> np.ndarray:
        if fold is None:
            return np.arange(0, 0)  # placeholder; set before reset via fold
        return np.asarray(fold.test_indices, dtype=np.int64)

    def set_walk_forward_fold(
        self,
        fold: WalkForwardFold,
        *,
        mode: str = "test",
    ) -> None:
        """
        Bind the environment to a purged walk-forward fold.

        mode='test'  → episode steps only on fold.test_indices
        mode='train' → episode steps only on purged fold.train_indices
        """
        if mode not in {"test", "train"}:
            raise ValueError("mode must be 'test' or 'train'")

        if mode == "test":
            self._active_indices = np.asarray(fold.test_indices, dtype=np.int64)
        else:
            n_samples = self._embeddings.shape[0]
            candidate = np.arange(0, int(fold.test_indices.min()), dtype=np.int64)
            self._active_indices = purge_indices(
                candidate,
                fold.test_indices,
                n_samples=n_samples,
                embargo=self.purge_embargo,
            )
        self._fold = fold

    @staticmethod
    def build_walk_forward_folds(
        n_samples: int,
        n_splits: int = 5,
        embargo: int = 5,
        min_train_size: int = 252,
        test_size: Optional[int] = 63,
    ) -> list[WalkForwardFold]:
        """Factory for purged walk-forward folds."""
        splitter = PurgedWalkForwardSplitter(
            n_splits=n_splits,
            embargo=embargo,
            min_train_size=min_train_size,
            test_size=test_size,
        )
        return list(splitter.split(n_samples))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def _obs_at(self, index: int) -> np.ndarray:
        return self._embeddings[index].astype(np.float64)

    def _information_ratio_component(self, excess: float) -> None:
        self._cumulative_excess += excess
        self._excess_sq_sum += excess * excess
        self._info_ratio_steps += 1

    def current_information_ratio(self) -> float:
        """Annualized IR proxy from accumulated excess returns (mean / std)."""
        if self._info_ratio_steps < 2:
            return 0.0
        mean = self._cumulative_excess / self._info_ratio_steps
        var = self._excess_sq_sum / self._info_ratio_steps - mean * mean
        std = float(np.sqrt(max(var, 1e-12)))
        return float(mean / std * np.sqrt(252.0))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "walk_forward_fold" in options:
            mode = options.get("mode", "test")
            self.set_walk_forward_fold(options["walk_forward_fold"], mode=mode)

        if self._active_indices.size == 0:
            n = self._embeddings.shape[0]
            self._active_indices = np.arange(n, dtype=np.int64)

        start = int(options.get("start_index", 0))
        if start < 0 or start >= self._active_indices.size:
            raise ValueError("start_index out of range for active fold")

        self._cursor = start
        self._step_in_episode = 0
        self._weights = np.ones(self._n_strategies, dtype=np.float64) / self._n_strategies
        self._portfolio_returns.clear()
        self._excess_returns.clear()
        self._cumulative_excess = 0.0
        self._excess_sq_sum = 0.0
        self._info_ratio_steps = 0

        index = int(self._active_indices[self._cursor])
        return self._obs_at(index), {
            "bar_index": index,
            "fold_id": None if self._fold is None else self._fold.fold_id,
            "weights": self._weights.copy(),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._cursor >= self._active_indices.size:
            raise RuntimeError("step() called on terminated episode; call reset()")

        index = int(self._active_indices[self._cursor])
        new_weights = self.softmax(np.asarray(action, dtype=np.float64))
        if self._settings is not None:
            from core.risk_manager import apply_kelly_from_settings

            new_weights = apply_kelly_from_settings(
                new_weights,
                self._strategy_matrix,
                self._settings,
                bar_index=index,
            )
        turnover = float(np.sum(np.abs(new_weights - self._weights)))

        leg_returns = self._strategy_matrix[index]
        portfolio_return = float(np.dot(new_weights, leg_returns))
        spy_return = float(self._spy[index])
        excess = portfolio_return - spy_return
        reward = excess - self.turnover_penalty_lambda * turnover

        self._weights = new_weights
        self._portfolio_returns.append(portfolio_return)
        self._excess_returns.append(excess)
        self._information_ratio_component(excess)

        self._cursor += 1
        self._step_in_episode += 1

        terminated = self._cursor >= self._active_indices.size
        truncated = False
        if self.episode_length is not None and self._step_in_episode >= self.episode_length:
            truncated = not terminated
            terminated = True

        if terminated:
            obs = self._obs_at(int(self._active_indices[-1]))
            next_index = int(self._active_indices[-1])
        else:
            next_index = int(self._active_indices[self._cursor])
            obs = self._obs_at(next_index)

        info = {
            "bar_index": index,
            "next_bar_index": next_index,
            "portfolio_return": portfolio_return,
            "spy_return": spy_return,
            "excess_return": excess,
            "turnover": turnover,
            "weights": new_weights.copy(),
            "information_ratio": self.current_information_ratio(),
            "fold_id": None if self._fold is None else self._fold.fold_id,
            "purge_embargo": self.purge_embargo,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return


def _synthetic_bundle(n_steps: int = 500, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, StrategyReturns, np.ndarray]:
    """Generate aligned synthetic data for smoke tests."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0, 1, size=(n_steps, dim))
    stat = rng.normal(0, 0.01, n_steps)
    vol = rng.normal(0, 0.012, n_steps)
    mr = rng.normal(0, 0.008, n_steps)
    spy = rng.normal(0.0002, 0.01, n_steps)
    return (
        embeddings,
        StrategyReturns(stat_arb=stat, vol_breakout=vol, mean_reversion=mr),
        spy,
    )


def run_walk_forward_smoke_test(
    n_steps: int = 500,
    n_splits: int = 3,
    embargo: int = 5,
) -> dict[str, Any]:
    """
    Exercise purged walk-forward binding and a short episode per fold.

    Returns summary dict with per-fold steps and terminal information ratio.
    """
    embeddings, strategies, spy = _synthetic_bundle(n_steps=n_steps)
    folds = TradingRoutingEnv.build_walk_forward_folds(
        n_samples=n_steps,
        n_splits=n_splits,
        embargo=embargo,
        min_train_size=100,
        test_size=50,
    )

    results: dict[str, Any] = {"folds": []}
    for fold in folds:
        env = TradingRoutingEnv(
            embeddings,
            strategies,
            spy,
            walk_forward_fold=fold,
            turnover_penalty_lambda=0.1,
            purge_embargo=embargo,
        )
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if truncated:
                break

        results["folds"].append(
            {
                "fold_id": fold.fold_id,
                "test_size": int(fold.test_indices.size),
                "train_size_purged": int(fold.train_indices.size),
                "steps": steps,
                "total_reward": total_reward,
                "information_ratio": info["information_ratio"],
                "purge_zone": (fold.purge_start, fold.purge_end),
            }
        )

    return results


if __name__ == "__main__":
    summary = run_walk_forward_smoke_test()
    print("Walk-forward smoke test")
    for fold in summary["folds"]:
        print(
            f"  fold {fold['fold_id']}: test={fold['test_size']} "
            f"train_purged={fold['train_size_purged']} "
            f"steps={fold['steps']} IR={fold['information_ratio']:.4f} "
            f"purge_zone={fold['purge_zone']}"
        )
