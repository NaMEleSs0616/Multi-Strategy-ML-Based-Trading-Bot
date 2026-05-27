"""SQLite OHLCV storage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, interval, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_interval ON bars(symbol, interval);
"""


class BarStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def upsert_bars(self, symbol: str, interval: str, bars: pd.DataFrame) -> int:
        if bars.empty:
            return 0
        frame = bars.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError(f"bars missing columns: {required - set(frame.columns)}")

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)

        rows = [
            (
                symbol.upper(),
                interval,
                ts.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
            for ts, row in frame.iterrows()
        ]

        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO bars (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, timestamp) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def load_bars(
        self,
        symbol: str,
        interval: str,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND interval = ?
        """
        params: list = [symbol.upper(), interval]
        if start is not None:
            query += " AND timestamp >= ?"
            params.append(pd.Timestamp(start).isoformat())
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(pd.Timestamp(end).isoformat())
        query += " ORDER BY timestamp"

        with self.connection() as conn:
            frame = pd.read_sql_query(query, conn, params=params)

        if frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Stored timestamps may now include explicit timezone offsets
        # (e.g. 2026-05-27T09:30:00-04:00) from Alpaca intraday syncs.
        # Parse as ISO8601 + UTC to handle mixed offset/naive strings safely.
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            format="ISO8601",
            utc=True,
            errors="coerce",
        )
        # Drop malformed timestamps defensively instead of crashing the pipeline.
        frame = frame.dropna(subset=["timestamp"])
        # Downstream code expects a plain DatetimeIndex.
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(None)
        frame = frame.set_index("timestamp")
        return frame.astype(float)

    def symbols(self, interval: Optional[str] = None) -> list[str]:
        query = "SELECT DISTINCT symbol FROM bars"
        params: list = []
        if interval:
            query += " WHERE interval = ?"
            params.append(interval)
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return sorted({r[0] for r in rows})
