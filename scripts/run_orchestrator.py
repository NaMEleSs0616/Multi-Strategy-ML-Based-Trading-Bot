#!/usr/bin/env python3
"""
CLI: three-stage Global-to-Local training orchestrator.

Stage 1 — global xLSTM (AdamW + OU SDE loss) → global_xlstm_weights.pt
Stage 2 — base PPO + purged walk-forward → base_ppo_router.zip
Stage 3 — async per-ticker fine-tune (action_net + log_std only) → {TICKER}_active_policy.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings_store import load_settings  # noqa: E402
from pipeline.orchestrator_provider import SettingsFeatureProvider  # noqa: E402
from pipeline.train_orchestrator import (  # noqa: E402
    Stage1Config,
    Stage2Config,
    Stage3Config,
    main as run_orchestrator,
)


def _stage_configs(settings: dict, input_dim: int) -> tuple[Stage1Config, Stage2Config, Stage3Config]:
    xlstm = settings.get("xlstm", {})
    ppo = settings.get("ppo", {})
    rl = settings.get("rl", {})
    wf = settings.get("walk_forward", {})

    stage1 = Stage1Config(
        input_dim=input_dim,
        hidden_dim=int(xlstm.get("hidden_dim", 128)),
        embedding_dim=int(rl.get("embedding_dim", 64)),
        num_blocks=int(xlstm.get("num_blocks", 2)),
        dropout=float(xlstm.get("dropout", 0.1)),
        sequence_length=int(xlstm.get("sequence_length", 60)),
        batch_size=int(xlstm.get("batch_size", 64)),
        learning_rate=float(xlstm.get("learning_rate", 1e-3)),
        weight_decay=float(xlstm.get("weight_decay", 1e-4)),
        max_epochs=int(xlstm.get("max_epochs", 100)),
        plateau_patience=int(xlstm.get("plateau_patience", 5)),
        plateau_min_delta=float(xlstm.get("plateau_min_delta", 1e-4)),
        val_fraction=float(xlstm.get("val_fraction", 0.2)),
        ou_gamma=float(xlstm.get("ou_gamma", 0.1)),
        ou_dt=float(xlstm.get("ou_dt", 1.0)),
    )
    stage2 = Stage2Config(
        n_splits=int(wf.get("n_splits", 5)),
        embargo=int(rl.get("purge_embargo_bars", 5)),
        min_train_size=int(wf.get("min_train_size", 252)),
        test_size=int(wf.get("test_size", 63)),
        total_timesteps_per_fold=int(ppo.get("total_timesteps", 20_000)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        n_steps=int(ppo.get("n_steps", 256)),
        batch_size=int(ppo.get("batch_size", 64)),
        turnover_penalty_lambda=float(rl.get("turnover_penalty_lambda", 0.1)),
    )
    stage3 = Stage3Config(
        epochs=int(settings.get("orchestrator", {}).get("finetune_epochs", 20)),
        n_steps_per_epoch=int(settings.get("orchestrator", {}).get("finetune_steps_per_epoch", 1024)),
        learning_rate=float(settings.get("orchestrator", {}).get("finetune_lr", 1e-3)),
    )
    return stage1, stage2, stage3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Global-to-Local 3-stage training (xLSTM → PPO → per-ticker fine-tune)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models" / "orchestrator",
        help="Output directory for global_xlstm_weights.pt, base_ppo_router.zip, ticker_policies/",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers for Stage 3 fine-tune (default: universe.equities from settings)",
    )
    parser.add_argument(
        "--global-symbol",
        default=None,
        help="Primary symbol for Stage 1/2 global training (default: first universe equity)",
    )
    parser.add_argument(
        "--no-yfinance",
        action="store_true",
        help="Synthetic global features for Stage 1; strategy bank still uses SQLite when available",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device override: cpu, cuda, or mps",
    )
    parser.add_argument(
        "--max-concurrent-ft",
        type=int,
        default=1,
        help="Max concurrent Stage-3 GPU jobs (default: 1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO logging from the orchestrator",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    provider = SettingsFeatureProvider(
        settings=settings,
        use_yfinance=not args.no_yfinance,
        global_symbol=args.global_symbol,
    )
    tickers = args.tickers or list(settings.get("universe", {}).get("equities", []))
    if not tickers:
        parser.error("No tickers: pass --tickers or set universe.equities in settings.yaml")

    stage1, stage2, stage3 = _stage_configs(settings, provider.input_dim)

    print(f"device={args.device or 'auto'} out_dir={args.out_dir}")
    print(f"global_symbol={provider.primary_symbol} stage3_tickers={tickers}")
    print(f"input_dim={provider.input_dim} embedding_dim={stage1.embedding_dim}")

    results = run_orchestrator(
        provider=provider,
        stage1_config=stage1,
        stage2_config=stage2,
        stage3_config=stage3,
        env_settings=settings,
        out_dir=args.out_dir,
        tickers=tickers,
        prefer_device=args.device,
        max_concurrent_ft=args.max_concurrent_ft,
    )

    print("\nArtifacts:")
    print(f"  global encoder: {args.out_dir / 'global_xlstm_weights.pt'}")
    print(f"  base PPO:       {args.out_dir / 'base_ppo_router.zip'}")
    for ticker, path in results.items():
        print(f"  {ticker}: {path}")


if __name__ == "__main__":
    main()
