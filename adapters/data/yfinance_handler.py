"""
YFinance-backed :class:`AbstractDataHandler` wrapping the local harvester stack.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from config.settings_store import PROJECT_ROOT
from core.interfaces import AbstractDataHandler, BarQuery, FeatureRequest, SyncReport
from data.features.macro_fred import fetch_macro_frame
from data.features.pit_features import build_pit_feature_frame
from data.features.sentiment_edgar import daily_sentiment_series
from data.harvester.storage import BarStore
from data.harvester.yfinance_loader import fetch_yfinance_bars


class YFinanceDataHandler(AbstractDataHandler):
    """OHLCV + PiT features via SQLite cache and yfinance fallback."""

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

    def _period_for_interval(self, interval: str) -> str:
        if interval in {"1d", "1wk"}:
            return str(self._period)
        return str(self._intraday_period)

    def fetch_bars(self, query: BarQuery) -> pd.DataFrame:
        bars = self._store.load_bars(query.symbol, query.interval)
        if len(bars) >= 30 and query.start is None and query.period is None:
            return bars

        period = query.period or self._period_for_interval(query.interval)
        start_s = query.start.isoformat() if query.start else None
        end_s = query.end.isoformat() if query.end else None
        fetched = fetch_yfinance_bars(
            query.symbol,
            interval=query.interval,
            period=None if start_s else period,
            start=start_s,
            end=end_s,
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
            raise RuntimeError(f"No bars for {primary}")

        pit_shift = int(request.pit_shift)
        frame = build_pit_feature_frame(
            bars,
            pit_shift=pit_shift,
            autocorr_window=int(self._feat_cfg.get("autocorr_window", 20)),
            atr_window=int(self._feat_cfg.get("atr_window", 14)),
            frac_diff_d=float(self._feat_cfg.get("frac_diff_d", 0.4)),
            frac_diff_thresh=float(self._feat_cfg.get("frac_diff_thresh", 1e-3)),
        )
        idx = pd.DatetimeIndex(frame.index).normalize()

        if request.include_macro or self._feat_cfg.get("include_macro", False):
            macro = fetch_macro_frame(idx, pit_shift=pit_shift)
            frame = frame.join(macro, how="left")

        if request.include_sentiment or self._feat_cfg.get("include_sentiment", False):
            sent = daily_sentiment_series(primary, idx, pit_shift=pit_shift)
            frame = frame.join(sent, how="left")

        return frame.dropna(how="any")

    def sync_universe(self, symbols: Sequence[str], intervals: Sequence[str]) -> SyncReport:
        entries: list[dict[str, Any]] = []
        for symbol in symbols:
            for interval in intervals:
                period = self._period_for_interval(interval)
                try:
                    bars = fetch_yfinance_bars(symbol, interval=interval, period=period)
                except Exception:
                    continue
                if bars.empty:
                    continue
                rows = self._store.upsert_bars(symbol, interval, bars)
                entries.append({"symbol": symbol, "interval": interval, "rows": rows})
        return SyncReport(entries=entries)

    @property
    def store(self) -> BarStore:
        return self._store
