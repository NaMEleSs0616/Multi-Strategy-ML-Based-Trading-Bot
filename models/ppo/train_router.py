"""
PPO capital router (Stage 2) on frozen xLSTM embeddings + live strategy bank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config.settings_store import PROJECT_ROOT, load_settings
from core.paths import ArtifactPaths
from pipeline.training_data import build_training_data
from rl.gym_trading_env import TradingRoutingEnv
from rl.validation.purge import PurgedWalkForwardSplitter


@dataclass
class PPORouterResult:
    model_path: Path
    folds_trained: int
    summary_path: Path
    n_timesteps: int


def train_ppo_router(
    settings: Optional[dict[str, Any]] = None,
    *,
    require_encoder: bool = False,
    fold_id: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> PPORouterResult:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError("Install stable-baselines3: pip install stable-baselines3") from exc

    settings = settings or load_settings()
    ppo_cfg = settings.setdefault(
        "ppo",
        {
            "total_timesteps": 20_000,
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "checkpoint_dir": "models/ppo/checkpoints",
        },
    )
    rl_cfg = settings["rl"]
    wf_cfg = settings["walk_forward"]

    data = build_training_data(settings, require_encoder=require_encoder)
    n_steps = data.embeddings.shape[0]

    splitter = PurgedWalkForwardSplitter(
        n_splits=int(wf_cfg.get("n_splits", 5)),
        embargo=int(rl_cfg["purge_embargo_bars"]),
        min_train_size=int(wf_cfg.get("min_train_size", 252)),
        test_size=int(wf_cfg.get("test_size", 63)),
    )
    folds = list(splitter.split(n_steps))
    if not folds:
        raise ValueError("No walk-forward folds available for PPO training")

    if fold_id is not None:
        folds = [f for f in folds if f.fold_id == fold_id]
        if not folds:
            raise ValueError(f"fold_id {fold_id} not found")

    paths = ArtifactPaths.from_settings(settings)
    if out_dir is not None:
        paths = ArtifactPaths(root=Path(out_dir), ticker_policies_subdir=paths.ticker_policies_subdir)
    paths.ensure_dirs()
    summary: list[dict[str, Any]] = []

    model = None
    for fold in folds:
        def make_env() -> TradingRoutingEnv:
            return TradingRoutingEnv(
                data.embeddings,
                data.strategy_returns,
                data.spy_returns,
                walk_forward_fold=fold,
                turnover_penalty_lambda=float(rl_cfg["turnover_penalty_lambda"]),
                turnover_penalty_multiplier=float(
                    rl_cfg.get("turnover_penalty_multiplier", 1.0)
                ),
                sortino_weight=float(rl_cfg.get("sortino_weight", 0.5)),
                outperformance_weight=float(rl_cfg.get("outperformance_weight", 0.5)),
                sortino_window=int(rl_cfg.get("sortino_window", 30)),
                purge_embargo=int(rl_cfg["purge_embargo_bars"]),
                settings=settings,
            )

        vec_env = DummyVecEnv([make_env])
        timesteps = int(ppo_cfg.get("total_timesteps", 20_000))

        if model is None:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
                n_steps=int(ppo_cfg.get("n_steps", 256)),
                batch_size=int(ppo_cfg.get("batch_size", 64)),
                verbose=0,
            )
        else:
            model.set_env(vec_env)

        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        summary.append(
            {
                "fold_id": fold.fold_id,
                "test_bars": int(fold.test_indices.size),
                "timesteps": timesteps,
            }
        )

    assert model is not None
    model_path = paths.ppo_router
    model.save(str(model_path))

    summary_path = paths.ppo_summary
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return PPORouterResult(
        model_path=model_path,
        folds_trained=len(summary),
        summary_path=summary_path,
        n_timesteps=n_steps,
    )


if __name__ == "__main__":
    result = train_ppo_router(require_encoder=False)
    print(f"Saved PPO router to {result.model_path} ({result.folds_trained} folds, T={result.n_timesteps})")
