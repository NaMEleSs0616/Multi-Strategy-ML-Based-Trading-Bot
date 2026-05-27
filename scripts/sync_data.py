#!/usr/bin/env python3
"""Download universe OHLCV into SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings_store import load_settings
from data.harvester.pipeline import sync_universe


def main() -> None:
    settings = load_settings()
    provider = str(settings.get("data", {}).get("provider", "yfinance"))
    print(f"provider={provider}")
    results = sync_universe(settings=settings)
    for row in results:
        print(f"{row.symbol:6s} {row.interval:4s}  {row.rows:5d} rows")
    print(f"Done — {len(results)} series synced.")


if __name__ == "__main__":
    main()
