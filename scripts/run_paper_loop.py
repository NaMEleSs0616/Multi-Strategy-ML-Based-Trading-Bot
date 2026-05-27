#!/usr/bin/env python3
"""CLI for offline paper-trading dry run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.paper_trading_loop import run_paper_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper loop dry run (no Alpaca keys)")
    parser.add_argument("--equal-weight", action="store_true", help="Skip PPO checkpoint")
    parser.add_argument("--symbol", default="SPY", help="Execution symbol")
    args = parser.parse_args()

    result = run_paper_loop(
        use_ppo=not args.equal_weight,
        dry_run=True,
        symbol=args.symbol,
    )
    print("Paper loop (dry run)")
    print(f"  Time:       {result.timestamp}")
    print(f"  Weights:    {result.weights}")
    print(f"  Exposure:   {result.gross_exposure:.3f}")
    print(f"  Equity:     {result.equity:,.2f}")
    print(f"  Orders:     {result.orders_submitted} submitted, {result.fills} fills")
    print(f"  Report:     {result.report_path}")


if __name__ == "__main__":
    main()
