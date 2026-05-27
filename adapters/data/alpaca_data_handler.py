"""Alpaca historical data handler with robust chunked backfill.

Goals:
- Support large intraday history pulls (multi-year 5m/15m) without oversized
  single requests.
- Keep downstream contract identical to :class:`AbstractDataHandler`
  (columns: ``open, high, low, close, volume``).
- Persist every successful fetch into :class:`BarStore` for local replay.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from config.settings_store import PROJECT_ROOT
from core.interfaces import AbstractDataHandler, BarQuery, FeatureRequest, SyncReport
from data.features.pit_features import build_pit_feature_frame
from data.harvester.alpaca_loader import period_to_timedelta, resolve_alpaca_range
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
        # Use explicit long lookbacks for Alpaca backfills.
        # Daily keeps the existing config knob; intraday defaults to 5y to
        # bypass provider-imposed short-window behavior from other vendors.
        self._period = str(data_cfg.get("yfinance_period", "2y"))
        self._intraday_period = str(data_cfg.get("intraday_period", "5y"))
        self._feat_cfg = settings.get("features", {})
        self._available = _alpaca_keys_present()
        self._api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        self._api_secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        # Data API endpoint itself is separate from paper/live trading endpoints,
        # but we still honor APCA_API_BASE_URL to choose intent and logging.
        self._base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        self._paper_mode = "paper" in self._base_url
        self._data_tz = str(data_cfg.get("timezone", "America/New_York"))
        self._sleep_s = float(data_cfg.get("alpaca_rate_limit_sleep_s", 0.25))
        self._chunk_days_intraday = int(data_cfg.get("alpaca_intraday_chunk_days", 30))
        self._chunk_days_daily = int(data_cfg.get("alpaca_daily_chunk_days", 365))
        self._client = None
        if not self._available:
            logger.info("Alpaca API keys not set — AlpacaDataHandler will return empty bars")
        else:
            self._client = self._create_client()
            logger.info(
                "AlpacaDataHandler initialized (base=%s, paper_mode=%s, intraday_period=%s)",
                self._base_url,
                self._paper_mode,
                self._intraday_period,
            )

    def _period_for_interval(self, interval: str) -> str:
        if interval in {"1d", "1wk"}:
            return str(self._period)
        return str(self._intraday_period)

    def _create_client(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError:
            logger.warning("alpaca-py not installed")
            return None
        # For historical data, Alpaca routes to data endpoint; we preserve
        # env-driven auth and avoid hardcoding secrets.
        return StockHistoricalDataClient(self._api_key, self._api_secret)

    @staticmethod
    def _timeframe(interval: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1d": TimeFrame.Day,
            "1h": TimeFrame.Hour,
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
        }
        return tf_map.get(interval)

    def _chunk_days_for_interval(self, interval: str) -> int:
        return self._chunk_days_daily if interval in {"1d", "1wk"} else self._chunk_days_intraday

    def _normalize_alpaca_frame(self, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize Alpaca response into a BarStore-compatible OHLCV frame.

        The pipeline contract expects lowercase columns; for compatibility with
        external tooling that refers to OHLCV in title case, we normalize from
        either representation.
        """
        if bars.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)

        frame = bars.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]
        out = frame[keep].copy()
        if len(keep) < 5:
            missing = {"open", "high", "low", "close", "volume"} - set(keep)
            raise ValueError(f"Alpaca bars missing required columns: {missing}")

        idx = pd.to_datetime(out.index, utc=True)
        # Keep timezone-aware index as requested; standardize to US/Eastern.
        idx = idx.tz_convert(self._data_tz)
        out.index = idx
        out = out.dropna(how="any")
        return out.sort_index()

    def fetch_historical_bars(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Robust historical fetch using chunked windows + retry sleeps.

        For large multi-year intraday pulls we request contiguous windows
        (default 30 days each) and concatenate while deduplicating timestamps.
        """
        if self._client is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        timeframe = self._timeframe(interval)
        if timeframe is None:
            logger.warning("Unsupported Alpaca interval %s", interval)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        from alpaca.data.requests import StockBarsRequest

        chunk_days = max(1, self._chunk_days_for_interval(interval))
        cursor = start
        parts: list[pd.DataFrame] = []

        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=cursor,
                end=chunk_end,
                adjustment="raw",
                feed="iex",
            )
            try:
                df = self._client.get_stock_bars(req).df
            except Exception as exc:
                # Soft backoff for intermittent rate limits / transient upstream issues.
                logger.warning(
                    "Alpaca chunk fetch failed (%s %s→%s): %s; sleeping %.2fs",
                    symbol,
                    cursor.isoformat(),
                    chunk_end.isoformat(),
                    exc,
                    self._sleep_s,
                )
                time.sleep(self._sleep_s)
                cursor = chunk_end
                continue

            if not df.empty:
                norm = self._normalize_alpaca_frame(df, symbol)
                if not norm.empty:
                    parts.append(norm)

            # Small sleep to avoid tripping burst limits when traversing 5y.
            time.sleep(self._sleep_s)
            cursor = chunk_end

        if not parts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = pd.concat(parts, axis=0)
        out = out[~out.index.duplicated(keep="last")]
        return out.sort_index()

    def fetch_bars(self, query: BarQuery) -> pd.DataFrame:
        if not self._available or self._client is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        symbol = query.symbol.upper()
        bars = self._store.load_bars(symbol, query.interval)
        if len(bars) >= 30 and query.start is None and query.period is None:
            return bars

        period = query.period or self._period_for_interval(query.interval)
        start_dt, end_dt = resolve_alpaca_range(
            start=query.start,
            end=query.end,
            period=None if query.start else period,
            default_period="2y" if query.interval in {"1d", "1wk"} else self._intraday_period,
        )
        fetched = self.fetch_historical_bars(symbol, query.interval, start=start_dt, end=end_dt)
        if not fetched.empty:
            self._store.upsert_bars(symbol, query.interval, fetched)
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
                start_dt, end_dt = resolve_alpaca_range(
                    period=period,
                    default_period="2y" if interval in {"1d", "1wk"} else self._intraday_period,
                )
                bars = self.fetch_historical_bars(
                    symbol.upper(),
                    interval,
                    start=start_dt,
                    end=end_dt,
                )
                if bars.empty:
                    continue
                rows = self._store.upsert_bars(symbol.upper(), interval, bars)
                entries.append({"symbol": symbol.upper(), "interval": interval, "rows": rows})
        return SyncReport(entries=entries)

    @property
    def store(self) -> BarStore:
        return self._store
