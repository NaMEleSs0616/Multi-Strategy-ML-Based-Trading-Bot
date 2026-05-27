"""Load and persist project settings from config/settings.yaml."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY

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
        "enable_daily_momentum": True,
        "daily_momentum_fast_window": 50,
        "daily_momentum_slow_window": 200,
        "enable_daily_trend": True,
        "daily_trend_fast_window": 50,
        "daily_trend_slow_window": 200,
        "enable_vix_fade": True,
        "vix_fade_threshold": 25.0,
        "vix_fade_roc_lookback": 1,
        "enable_cross_sectional_mom": True,
        "cross_sectional_mom_lookback": 90,
        "enable_fractional_stat_arb": True,
        "fractional_stat_arb_window": 252,
        "fractional_stat_arb_entry_z": 2.0,
        "fractional_stat_arb_exit_z": 0.5,
        "fractional_stat_arb_gph_m": 10,
        "fractional_stat_arb_d_max": 0.5,
        "enable_vrp_harvesting": True,
        "vrp_rv_window": 30,
        "vrp_z_window": 252,
        "vrp_z_threshold": 2.0,
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
        "max_single_weight_live": 0.60,
        "min_momentum_allocation": 0.25,
        "momentum_win_rate_threshold": 0.55,
        "momentum_boost": 1.5,
    },
    "rl": {
        "embedding_dim": 64,
        "n_strategies": 9,
        "turnover_penalty_bps": 2.0,
        "turnover_weight_change_threshold": 0.05,
        "turnover_penalty_lambda": 0.1,
        "turnover_penalty_multiplier": 0.5,
        "sortino_weight": 0.5,
        "outperformance_weight": 0.5,
        "sortino_window": 30,
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
    "artifacts": {
        "root_dir": "models/checkpoints",
        "ticker_policies_subdir": "ticker_policies",
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


def _apply_env_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Force environment variables to win over YAML for the small set of toggles
    that have to disable expensive optional features in CI / Docker.

    The mutation is intentional: every downstream consumer should see the
    same dict regardless of which entry point loaded it.

    Recognized variables:
      - SKIP_EDGAR_SENTIMENT  → features.include_sentiment = False
      - SKIP_FRED_MACRO       → features.include_macro     = False
    """
    feats = settings.setdefault("features", {})

    if _is_truthy(os.environ.get("SKIP_EDGAR_SENTIMENT")):
        prev = feats.get("include_sentiment", None)
        feats["include_sentiment"] = False
        if prev:
            logger.info(
                "SKIP_EDGAR_SENTIMENT is set; forcing features.include_sentiment=False"
            )

    if _is_truthy(os.environ.get("SKIP_FRED_MACRO")):
        prev = feats.get("include_macro", None)
        feats["include_macro"] = False
        if prev:
            logger.info(
                "SKIP_FRED_MACRO is set; forcing features.include_macro=False"
            )

    return settings


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load settings YAML, fill missing keys from DEFAULTS, apply env overrides."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        return _apply_env_overrides(deepcopy(DEFAULTS))

    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged = _deep_merge(DEFAULTS, raw)
    return _apply_env_overrides(merged)


def save_settings(settings: dict[str, Any], path: Path | None = None) -> Path:
    """Persist settings to YAML."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False, default_flow_style=False)
    return settings_path
