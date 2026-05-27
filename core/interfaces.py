"""
Enterprise modularity layer — abstract data and execution contracts.

All training, backtesting, and routing code must depend only on these ABCs,
never on broker- or vendor-specific implementations.

Concrete handlers (post-review wiring):
  - ``adapters.data.yfinance_handler.YFinanceDataHandler``
  - ``adapters.data.alpaca_data_handler.AlpacaDataHandler``  (future)
  - ``adapters.execution.alpaca_execution.AlpacaExecutionHandler``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Shared value types (broker-agnostic)
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    # Market orders are intentionally omitted — passive limits only.


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class BarQuery:
    """OHLCV request specification."""

    symbol: str
    interval: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    period: Optional[str] = None  # vendor-specific lookback, e.g. "2y", "60d"


@dataclass(frozen=True)
class PassiveQuote:
    """Top-of-book snapshot for limit-order pricing."""

    symbol: str
    bid: float
    ask: float
    timestamp: Optional[datetime] = None


@dataclass(frozen=True)
class LimitOrderRequest:
    """Passive limit order (liquidity-providing)."""

    symbol: str
    side: OrderSide
    quantity: float
    limit_price: float
    client_order_id: Optional[str] = None


@dataclass
class OrderAck:
    """Broker acknowledgement for a submitted order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    limit_price: float
    status: OrderStatus = OrderStatus.ACCEPTED
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FillEvent:
    """Execution fill notification."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureRequest:
    """
    Feature-matrix request for xLSTM / PiT pipelines.

    Implementations must return PiT-shifted columns only (no contemporaneous leakage).
    """

    symbols: Sequence[str]
    daily_interval: str = "1d"
    pit_shift: int = 1
    include_macro: bool = False
    include_sentiment: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncReport:
    """Result of a universe sync operation."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return int(sum(e.get("rows", 0) for e in self.entries))


# ---------------------------------------------------------------------------
# Abstract data plane
# ---------------------------------------------------------------------------


class AbstractDataHandler(ABC):
    """
    Vendor-neutral market and feature data access.

    Responsibilities
    ----------------
    - Historical / live OHLCV
    - Optional macro & sentiment enrichment (Phase 2)
    - PiT-safe feature frames for the xLSTM encoder
    """

    @abstractmethod
    def fetch_bars(self, query: BarQuery) -> pd.DataFrame:
        """
        Return OHLCV bars indexed by timestamp.

        Columns: open, high, low, close, volume (lowercase).
        """

    @abstractmethod
    def fetch_feature_frame(self, request: FeatureRequest) -> pd.DataFrame:
        """
        Return a PiT-compliant feature matrix aligned to the primary symbol timeline.

        Index: DatetimeIndex (normalized). Columns: model inputs including any
        macro / sentiment fields when requested.
        """

    @abstractmethod
    def sync_universe(self, symbols: Sequence[str], intervals: Sequence[str]) -> SyncReport:
        """Persist latest vendor data locally (e.g. SQLite) for offline replay."""

    def fetch_bars_many(
        self,
        symbols: Sequence[str],
        interval: str,
        *,
        period: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Convenience wrapper; default loops ``fetch_bars``."""
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            out[symbol] = self.fetch_bars(
                BarQuery(symbol=symbol, interval=interval, period=period)
            )
        return out


# ---------------------------------------------------------------------------
# Abstract execution plane
# ---------------------------------------------------------------------------


class AbstractExecutionHandler(ABC):
    """
    Vendor-neutral order routing and account state.

    Hard constraint: **limit orders only** — implementations must reject market orders.
    """

    @abstractmethod
    def get_quote(self, symbol: str) -> PassiveQuote:
        """Current bid/ask for passive limit pricing."""

    @abstractmethod
    def submit_limit_order(self, request: LimitOrderRequest) -> OrderAck:
        """
        Submit a passive limit order.

        Raises
        ------
        ValueError
            If the request implies a non-limit order type.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation was accepted."""

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderAck]:
        """List working orders, optionally filtered by symbol."""

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Map symbol → signed quantity (shares)."""

    @abstractmethod
    def get_equity(self) -> float:
        """Total account equity (cash + marked positions)."""

    def validate_limit_only(self, order_type: OrderType) -> None:
        if order_type != OrderType.LIMIT:
            raise ValueError(
                "Market orders are prohibited. The system uses passive LIMIT orders only."
            )
