"""Sync universe OHLCV into SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config.settings_store import load_settings
from adapters.factory import create_data_handler


@dataclass
class SyncResult:
    symbol: str
    interval: str
    rows: int


def sync_universe(
    settings: Optional[dict[str, Any]] = None,
    *,
    intervals: Optional[list[str]] = None,
) -> list[SyncResult]:
    settings = settings or load_settings()
    data_cfg = settings["data"]

    symbols = list(settings["universe"]["equities"]) + list(settings["universe"]["etfs"])
    if intervals is None:
        intervals = [data_cfg["daily_interval"], *data_cfg["intraday_intervals"]]

    handler = create_data_handler(settings)
    report = handler.sync_universe(symbols, intervals)
    return [
        SyncResult(symbol=e["symbol"], interval=e["interval"], rows=int(e["rows"]))
        for e in report.entries
    ]
