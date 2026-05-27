#!/usr/bin/env python3
"""CLI for walk-forward OOS report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.walk_forward_report import run_walk_forward_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS portfolio report")
    parser.add_argument("--equal-weight", action="store_true", help="Skip PPO checkpoint")
    parser.add_argument("--no-csv", action="store_true", help="Skip per-fold CSV export")
    parser.add_argument("--require-encoder", action="store_true", help="Require xLSTM checkpoint")
    args = parser.parse_args()

    result = run_walk_forward_report(
        use_ppo=not args.equal_weight,
        export_csv=not args.no_csv,
        require_encoder=args.require_encoder,
    )
    agg = result.aggregate
    print("Walk-forward OOS report")
    print(f"  Folds:              {agg['n_folds']}")
    print(f"  Policy:             {agg['policy']}")
    print(f"  Mean IR:            {agg['mean_information_ratio']:.4f}")
    print(f"  Mean total return:  {agg['mean_total_return']:.2%}")
    print(f"  Worst max DD:       {agg['worst_max_drawdown']:.2%}")
    print(f"  Mean turnover:      {agg['mean_avg_turnover']:.4f}")
    print(f"  JSON:               {result.report_path}")
    if result.csv_dir:
        print(f"  CSV dir:            {result.csv_dir}")


if __name__ == "__main__":
    main()
