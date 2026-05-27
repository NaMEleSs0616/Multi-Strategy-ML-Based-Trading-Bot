"""Load and persist project settings from config/settings.yaml."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

DEFAULTS: dict[str, Any] = {
    "universe": {
        "equities": ["NVDA", "AMD", "META", "GOOGL", "MSFT"],
        "etfs": ["QQQ", "SPY", "XLK"],
    },
    "data": {
        "provider": "yfinance",
        "sqlite_path": "data/storage/market_data.db",
        "daily_interval": "1d",
        "intraday_intervals": ["5m", "15m"],
        "yfinance_period": "2y",
        "intraday_period": "60d",
    },
    "strategies": {
        "pair": ["NVDA", "AMD"],
        "dynamic_pair_selection": False,
        "transaction_cost_bps": 5.0,
        "stat_arb_z_window": 20,
        "stat_arb_entry_z": 2.0,
        "vol_atr_window": 14,
        "vol_atr_mult": 1.5,
        "vol_lookback": 20,
        "mr_rsi_window": 14,
        "mr_bb_window": 20,
    },
    "features": {
        "frac_diff_d": 0.4,
        "frac_diff_thresh": 0.001,
        "autocorr_window": 20,
        "atr_window": 14,
        "pit_shift": 1,
        "include_macro": False,
        "include_sentiment": False,
    },
    "risk": {
        "enabled": True,
        "kelly_window": 30,
        "kelly_fraction": 0.5,
    },
    "rl": {
        "embedding_dim": 64,
        "n_strategies": 3,
        "turnover_penalty_lambda": 0.1,
        "purge_embargo_bars": 5,
    },
    "xlstm": {
        "hidden_dim": 128,
        "num_blocks": 2,
        "dropout": 0.1,
        "sequence_length": 60,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "val_fraction": 0.2,
        "plateau_patience": 5,
        "plateau_min_delta": 0.0001,
        "checkpoint_dir": "models/xlstm/checkpoints",
        "export_embeddings": True,
        "yfinance_period": "2y",
        "synthetic_bars": 2500,
        "ou_gamma": 0.1,
        "ou_dt": 1.0,
    },
    "ppo": {
        "total_timesteps": 20000,
        "learning_rate": 0.0003,
        "n_steps": 256,
        "batch_size": 64,
        "checkpoint_dir": "models/ppo/checkpoints",
    },
    "orchestrator": {
        "finetune_epochs": 20,
        "finetune_steps_per_epoch": 1024,
        "finetune_lr": 0.001,
    },
    "walk_forward": {
        "n_splits": 5,
        "min_train_size": 252,
        "test_size": 63,
        "smoke_n_steps": 500,
    },
    "execution": {
        "mode": "backtest",
        "paper": True,
        "order_type": "limit",
        "latency_bars": 1,
        "spread_bps": 5.0,
        "allow_market_orders": False,
    },
    "diagnostics": {
        "audit_n_bars": 120,
        "audit_seed": 42,
    },
    "backtest": {
        "initial_cash": 100_000.0,
        "rebalance_threshold": 0.05,
        "report_dir": "backtesting/reports",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load settings YAML, filling missing keys from DEFAULTS."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        return deepcopy(DEFAULTS)

    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULTS, raw)


def save_settings(settings: dict[str, Any], path: Path | None = None) -> Path:
    """Persist settings to YAML."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False, default_flow_style=False)
    return settings_path
