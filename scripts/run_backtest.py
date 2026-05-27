#!/usr/bin/env python3
"""CLI for routed portfolio backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtesting.run_backtest import run_portfolio_backtest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest PPO-routed strategy portfolio")
    parser.add_argument("--equal-weight", action="store_true", help="Skip PPO; use 1/3 each")
    parser.add_argument("--execution", action="store_true", help="Also run SPY limit-order simulation")
    args = parser.parse_args()

    result = run_portfolio_backtest(
        use_ppo=not args.equal_weight,
        run_execution=args.execution,
    )
    m = result.portfolio.metrics
    print("Portfolio backtest")
    print(f"  Information Ratio: {m['information_ratio']:.4f}")
    print(f"  Total return:      {m['total_return']:.2%}")
    print(f"  SPY return:        {m['spy_total_return']:.2%}")
    print(f"  Max drawdown:      {m['max_drawdown']:.2%}")
    print(f"  Avg turnover:      {m['avg_turnover']:.4f}")
    print(f"  Report:            {result.report_path}")


if __name__ == "__main__":
    main()
