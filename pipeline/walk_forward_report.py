"""
Walk-forward out-of-sample report: per-fold portfolio metrics and aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtesting.portfolio_simulator import PortfolioBacktestResult, simulate_routed_portfolio
from backtesting.ppo_router import equal_weight_fn, load_weight_fn
from config.settings_store import PROJECT_ROOT, load_settings
from pipeline.training_data import TrainingData, build_training_data
from rl.gym_trading_env import StrategyReturns
from rl.validation.purge import PurgedWalkForwardSplitter, WalkForwardFold


@dataclass
class WalkForwardReportResult:
    folds: list[dict[str, Any]]
    aggregate: dict[str, float]
    report_path: Path
    csv_dir: Optional[Path]


def _slice_training_data(data: TrainingData, indices: np.ndarray) -> TrainingData:
    idx = np.asarray(indices, dtype=np.int64)
    return TrainingData(
        embeddings=data.embeddings[idx],
        strategy_returns=data.strategy_returns.slice(idx),
        spy_returns=data.spy_returns[idx],
        index=pd.DatetimeIndex(data.index[idx]),
        features=data.features[idx],
    )


def _weight_fn_for_fold(
    data: TrainingData,
    fold: WalkForwardFold,
    *,
    use_ppo: bool,
    n_strategies: int,
) -> Callable[[int], np.ndarray]:
    if use_ppo:
        global_fn = load_weight_fn(data.embeddings)

        def fn(local_t: int) -> np.ndarray:
            global_t = int(fold.test_indices[local_t])
            return global_fn(global_t)

        return fn

    base = equal_weight_fn(n_strategies)

    def fn_eq(local_t: int) -> np.ndarray:
        return base(local_t)

    return fn_eq


def _eval_fold(
    data: TrainingData,
    fold: WalkForwardFold,
    settings: dict[str, Any],
    *,
    use_ppo: bool,
) -> dict[str, Any]:
    sliced = _slice_training_data(data, fold.test_indices)
    n_strategies = int(settings["rl"].get("n_strategies", 3))
    weight_fn = _weight_fn_for_fold(
        data,
        fold,
        use_ppo=use_ppo,
        n_strategies=n_strategies,
    )
    rl_cfg = settings["rl"]
    result = simulate_routed_portfolio(
        sliced,
        weight_fn,
        turnover_penalty_lambda=float(rl_cfg.get("turnover_penalty_lambda", 0.1)),
        apply_penalty_to_returns=False,
        settings=settings,
    )
    return {
        "fold_id": fold.fold_id,
        "test_bars": len(fold.test_indices),
        "train_bars": len(fold.train_indices),
        "test_start": str(sliced.index[0]) if len(sliced.index) else None,
        "test_end": str(sliced.index[-1]) if len(sliced.index) else None,
        "metrics": result.metrics,
        "portfolio_result": result,
    }


def run_walk_forward_report(
    settings: Optional[dict[str, Any]] = None,
    *,
    use_ppo: bool = True,
    equal_weight_fallback: bool = True,
    export_csv: bool = True,
    require_encoder: bool = False,
) -> WalkForwardReportResult:
    settings = settings or load_settings()
    wf_cfg = settings.setdefault("walk_forward", {})
    rl_cfg = settings["rl"]

    data = build_training_data(settings, require_encoder=require_encoder)
    n_steps = data.embeddings.shape[0]

    splitter = PurgedWalkForwardSplitter(
        n_splits=int(wf_cfg.get("n_splits", 5)),
        embargo=int(rl_cfg.get("purge_embargo_bars", 5)),
        min_train_size=int(wf_cfg.get("min_train_size", 252)),
        test_size=int(wf_cfg.get("test_size", 63)),
    )
    folds = list(splitter.split(n_steps))
    if not folds:
        raise ValueError("No walk-forward folds available for OOS report")

    ppo_ok = use_ppo
    if use_ppo:
        try:
            from backtesting.ppo_router import resolve_ppo_path

            resolve_ppo_path(settings)
        except FileNotFoundError:
            if not equal_weight_fallback:
                raise
            ppo_ok = False

    fold_rows: list[dict[str, Any]] = []
    portfolio_results: list[PortfolioBacktestResult] = []

    for fold in folds:
        row = _eval_fold(data, fold, settings, use_ppo=ppo_ok)
        portfolio_results.append(row.pop("portfolio_result"))
        fold_rows.append(row)

    irs = [r["metrics"]["information_ratio"] for r in fold_rows]
    rets = [r["metrics"]["total_return"] for r in fold_rows]
    dds = [r["metrics"]["max_drawdown"] for r in fold_rows]
    turns = [r["metrics"]["avg_turnover"] for r in fold_rows]

    aggregate = {
        "mean_information_ratio": float(np.mean(irs)) if irs else 0.0,
        "mean_total_return": float(np.mean(rets)) if rets else 0.0,
        "worst_max_drawdown": float(np.min(dds)) if dds else 0.0,
        "mean_avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "n_folds": len(fold_rows),
        "policy": "ppo" if ppo_ok else "equal_weight",
    }

    report_dir = PROJECT_ROOT / settings.get("backtest", {}).get(
        "report_dir", "backtesting/reports"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "walk_forward_report.json"

    payload = {
        "aggregate": aggregate,
        "folds": [
            {k: v for k, v in row.items() if k != "portfolio_result"}
            for row in fold_rows
        ],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_dir: Optional[Path] = None
    if export_csv:
        csv_dir = report_dir / "walk_forward_folds"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for fold_row, port in zip(fold_rows, portfolio_results):
            fid = fold_row["fold_id"]
            out = pd.DataFrame(
                {
                    "portfolio_return": port.portfolio_returns,
                    "spy_return": port.spy_returns,
                    "turnover": port.turnover,
                },
                index=port.index,
            )
            out.to_csv(csv_dir / f"fold_{fid}.csv")

    return WalkForwardReportResult(
        folds=fold_rows,
        aggregate=aggregate,
        report_path=report_path,
        csv_dir=csv_dir,
    )
