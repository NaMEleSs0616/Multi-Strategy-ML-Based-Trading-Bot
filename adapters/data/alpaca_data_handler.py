"""
Alpaca historical data handler (minimal; requires API keys).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional, Sequence

import pandas as pd

from core.interfaces import AbstractDataHandler, BarQuery, FeatureRequest, SyncReport
from data.features.pit_features import build_pit_feature_frame

logger = logging.getLogger(__name__)


def _alpaca_keys_present() -> bool:
    return bool(os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    )


class AlpacaDataHandler(AbstractDataHandler):
    """
    Stub/minimal Alpaca historical bars when credentials are configured.

    Falls back to empty frames when keys are missing (callers should prefer
  ``YFinanceDataHandler`` via the factory).
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings = settings
        self._feat_cfg = settings.get("features", {})
        self._available = _alpaca_keys_present()
        if not self._available:
            logger.info("Alpaca API keys not set — AlpacaDataHandler will return empty bars")

    def fetch_bars(self, query: BarQuery) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            logger.warning("alpaca-py not installed")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret)

        tf_map = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour, "15m": TimeFrame.Minute15}
        timeframe = tf_map.get(query.interval, TimeFrame.Day)

        req = StockBarsRequest(
            symbol_or_symbols=query.symbol,
            timeframe=timeframe,
            start=query.start,
            end=query.end,
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(query.symbol, level=0)
        bars = bars.rename(columns=str.lower)
        out = bars[["open", "high", "low", "close", "volume"]].copy()
        out.index = pd.to_datetime(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        return out.sort_index()

    def fetch_feature_frame(self, request: FeatureRequest) -> pd.DataFrame:
        if not request.symbols:
            raise ValueError("FeatureRequest.symbols must be non-empty")
        bars = self.fetch_bars(
            BarQuery(symbol=str(request.symbols[0]), interval=request.daily_interval)
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
        )

    def sync_universe(self, symbols: Sequence[str], intervals: Sequence[str]) -> SyncReport:
        entries: list[dict[str, Any]] = []
        for symbol in symbols:
            for interval in intervals:
                bars = self.fetch_bars(BarQuery(symbol=symbol, interval=interval))
                if not bars.empty:
                    entries.append({"symbol": symbol, "interval": interval, "rows": len(bars)})
        return SyncReport(entries=entries)
