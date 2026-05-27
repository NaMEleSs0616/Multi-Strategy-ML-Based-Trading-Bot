"""
Alpaca historical data handler (SQLite cache + Alpaca Market Data API).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from config.settings_store import PROJECT_ROOT
from core.interfaces import AbstractDataHandler, BarQuery, FeatureRequest, SyncReport
from data.features.pit_features import build_pit_feature_frame
from data.harvester.alpaca_loader import fetch_alpaca_bars
from data.harvester.storage import BarStore

logger = logging.getLogger(__name__)


def _alpaca_keys_present() -> bool:
    return bool(os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    )


class AlpacaDataHandler(AbstractDataHandler):
    """OHLCV + PiT features via SQLite cache and Alpaca fallback."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        db_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings
        data_cfg = settings.get("data", {})
        self._db_path = db_path or (PROJECT_ROOT / data_cfg.get("sqlite_path", "data/storage/market_data.db"))
        self._store = BarStore(self._db_path)
        self._period = data_cfg.get("yfinance_period", "2y")
        self._intraday_period = data_cfg.get("intraday_period", "60d")
        self._feat_cfg = settings.get("features", {})
        self._available = _alpaca_keys_present()
        if not self._available:
            logger.info("Alpaca API keys not set — AlpacaDataHandler will return empty bars")

    def _period_for_interval(self, interval: str) -> str:
        if interval in {"1d", "1wk"}:
            return str(self._period)
        return str(self._intraday_period)

    def fetch_bars(self, query: BarQuery) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        bars = self._store.load_bars(query.symbol, query.interval)
        if len(bars) >= 30 and query.start is None and query.period is None:
            return bars

        period = query.period or self._period_for_interval(query.interval)
        fetched = fetch_alpaca_bars(
            query.symbol,
            interval=query.interval,
            period=None if query.start else period,
            start=query.start,
            end=query.end,
        )
        if not fetched.empty:
            self._store.upsert_bars(query.symbol, query.interval, fetched)
        return fetched if not fetched.empty else bars

    def fetch_feature_frame(self, request: FeatureRequest) -> pd.DataFrame:
        if not request.symbols:
            raise ValueError("FeatureRequest.symbols must be non-empty")

        primary = str(request.symbols[0])
        bars = self.fetch_bars(
            BarQuery(
                symbol=primary,
                interval=request.daily_interval,
                period=self._settings.get("xlstm", {}).get("yfinance_period", self._period),
            )
        )
        if bars.empty:
            raise RuntimeError("Alpaca returned no bars — check API keys or use yfinance provider")

        return build_pit_feature_frame(
            bars,
            pit_shift=int(request.pit_shift),
            autocorr_window=int(self._feat_cfg.get("autocorr_window", 20)),
            atr_window=int(self._feat_cfg.get("atr_window", 14)),
            frac_diff_d=float(self._feat_cfg.get("frac_diff_d", 0.4)),
            frac_diff_thresh=float(self._feat_cfg.get("frac_diff_thresh", 1e-3)),
        ).dropna(how="any")

    def sync_universe(self, symbols: Sequence[str], intervals: Sequence[str]) -> SyncReport:
        entries: list[dict[str, Any]] = []
        for symbol in symbols:
            for interval in intervals:
                period = self._period_for_interval(interval)
                try:
                    bars = fetch_alpaca_bars(symbol, interval=interval, period=period)
                except Exception:
                    logger.exception("Alpaca sync failed for %s %s", symbol, interval)
                    continue
                if bars.empty:
                    continue
                rows = self._store.upsert_bars(symbol, interval, bars)
                entries.append({"symbol": symbol, "interval": interval, "rows": rows})
        return SyncReport(entries=entries)

    @property
    def store(self) -> BarStore:
        return self._store
