"""Load PPO policy and produce softmax allocation weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from config.settings_store import PROJECT_ROOT, load_settings
from rl.gym_trading_env import TradingRoutingEnv


def resolve_ppo_path(settings: Optional[dict[str, Any]] = None) -> Path:
    settings = settings or load_settings()
    path = PROJECT_ROOT / settings.get("ppo", {}).get(
        "checkpoint_dir", "models/ppo/checkpoints"
    ) / "ppo_router.zip"
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found at {path}. Train with scripts/train_ppo.py")
    return path


def load_weight_fn(
    embeddings: np.ndarray,
    *,
    model_path: Optional[Path] = None,
    deterministic: bool = True,
) -> Callable[[int], np.ndarray]:
    from stable_baselines3 import PPO

    path = model_path or resolve_ppo_path()
    model = PPO.load(str(path))

    def weights_at(t: int) -> np.ndarray:
        obs = embeddings[t].astype(np.float64)
        action, _ = model.predict(obs, deterministic=deterministic)
        return TradingRoutingEnv.softmax(np.asarray(action, dtype=np.float64))

    return weights_at


def wrap_weight_fn_with_kelly(
    weight_fn: Callable[[int], np.ndarray],
    strategy_matrix: np.ndarray,
    settings: dict[str, Any],
) -> Callable[[int], np.ndarray]:
    """Apply fractional Kelly cap at inference time (PPO or custom policies)."""
    from core.risk_manager import apply_kelly_from_settings

    def inner(t: int) -> np.ndarray:
        w = weight_fn(t)
        return apply_kelly_from_settings(w, strategy_matrix, settings, bar_index=t)

    return inner


def equal_weight_fn(n_strategies: int = 3) -> Callable[[int], np.ndarray]:
    w = np.ones(n_strategies, dtype=np.float64) / n_strategies

    def _fn(_t: int) -> np.ndarray:
        return w.copy()

    return _fn
