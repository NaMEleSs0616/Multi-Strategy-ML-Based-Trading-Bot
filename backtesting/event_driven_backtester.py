"""
Strictly event-driven backtester.

Iterates bar-by-bar via a priority-free FIFO event queue. Passive limit orders
only; market orders are rejected. Includes an automated look-ahead bias audit.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass(order=True)
class Event:
    """Queue item ordered by (timestamp, sequence) for deterministic replay."""

    priority: tuple[pd.Timestamp, int] = field(compare=True)
    event_type: EventType = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


@dataclass
class BarSnapshot:
    """Point-in-time OHLCV plus top-of-book for a single symbol at one bar."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bar_index: int


@dataclass
class LimitOrder:
    order_id: int
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: float
    limit_price: float
    created_at: pd.Timestamp
    active_after_bar: int
    status: str = "PENDING"


@dataclass
class FillRecord:
    order_id: int
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: pd.Timestamp
    bar_index: int


class StrategyHandler(Protocol):
    def on_bar(self, history: pd.DataFrame, bar: BarSnapshot) -> List[dict]:
        """Return zero or more signal dicts using only history through bar."""


class EventDrivenBacktester:
    """
    Event-driven simulation engine.

    Parameters
    ----------
    bars
        Multi-index (timestamp, symbol) or single-symbol DataFrame with columns
        open, high, low, close, volume. Optional bid, ask; if absent, synthetic
        spread around close is derived.
    strategy
        Callable invoked each market event with PiT history slice.
    initial_cash
        Starting capital.
    latency_bars
        Orders become eligible for matching after this many completed bars.
    max_ffill_bars
        Maximum consecutive forward-fills allowed inside a bounded window
        (unbounded ffill is treated as a bias violation in the audit).
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        strategy: StrategyHandler | Callable[[pd.DataFrame, BarSnapshot], List[dict]],
        *,
        initial_cash: float = 100_000.0,
        latency_bars: int = 1,
        max_ffill_bars: int = 0,
        spread_bps: float = 5.0,
    ) -> None:
        self._bars = self._normalize_bars(bars)
        self._strategy = strategy
        self.initial_cash = initial_cash
        self.latency_bars = latency_bars
        self.max_ffill_bars = max_ffill_bars
        self.spread_bps = spread_bps

        self._event_queue: List[Event] = []
        self._sequence = 0
        self._order_id = 0

        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.open_orders: Dict[int, LimitOrder] = {}
        self.fills: List[FillRecord] = []
        self.equity_curve: List[tuple[pd.Timestamp, float]] = []

        self._history_rows: Deque[dict] = deque(maxlen=None)
        self._current_bar: Optional[BarSnapshot] = None
        self._bar_index = -1

    @staticmethod
    def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        lower_cols = {c.lower(): c for c in bars.columns}
        rename = {lower_cols[k]: k for k in required if k in lower_cols}
        frame = bars.rename(columns=rename).copy()
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        return frame

    def _enqueue(self, event_type: EventType, timestamp: pd.Timestamp, payload: dict) -> None:
        self._sequence += 1
        heapq.heappush(
            self._event_queue,
            Event(
                priority=(timestamp, self._sequence),
                event_type=event_type,
                payload=payload,
            ),
        )

    def _synthetic_book(self, close: float) -> tuple[float, float]:
        half_spread = close * (self.spread_bps / 10_000.0) / 2.0
        return close - half_spread, close + half_spread

    def _bar_snapshot(self, row: pd.Series, timestamp: pd.Timestamp, symbol: str) -> BarSnapshot:
        close = float(row["close"])
        if "bid" in row and "ask" in row and pd.notna(row["bid"]) and pd.notna(row["ask"]):
            bid, ask = float(row["bid"]), float(row["ask"])
        else:
            bid, ask = self._synthetic_book(close)

        return BarSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=close,
            volume=float(row["volume"]),
            bid=bid,
            ask=ask,
            bar_index=self._bar_index,
        )

    def _history_frame(self) -> pd.DataFrame:
        if not self._history_rows:
            return pd.DataFrame()
        return pd.DataFrame(self._history_rows).set_index("timestamp")

    def _process_market(self, payload: dict) -> None:
        bar: BarSnapshot = payload["bar"]
        self._current_bar = bar
        self._history_rows.append(
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "bid": bar.bid,
                "ask": bar.ask,
            }
        )
        history = self._history_frame()

        signals = self._strategy(history, bar)
        for signal in signals:
            self._enqueue(EventType.SIGNAL, bar.timestamp, {"signal": signal, "bar": bar})

        self._match_open_orders(bar)
        self._mark_equity(bar.timestamp)

    def _process_signal(self, payload: dict) -> None:
        signal = payload["signal"]
        bar: BarSnapshot = payload["bar"]
        side = signal.get("side", "").upper()
        qty = float(signal.get("quantity", 0.0))
        if side not in {"BUY", "SELL"} or qty <= 0:
            return

        # Passive limit: buy at bid (provide liquidity), sell at ask.
        if side == "BUY":
            limit_price = bar.bid
        else:
            limit_price = bar.ask

        self.submit_limit_order(
            symbol=bar.symbol,
            side=side,
            quantity=qty,
            limit_price=limit_price,
            timestamp=bar.timestamp,
        )

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        timestamp: pd.Timestamp,
        *,
        order_type: str = "LIMIT",
    ) -> int:
        if order_type.upper() == "MARKET":
            raise ValueError("Market orders are prohibited; use passive LIMIT orders only.")
        if order_type.upper() != "LIMIT":
            raise ValueError(f"Unsupported order type: {order_type}")

        self._order_id += 1
        order = LimitOrder(
            order_id=self._order_id,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            limit_price=limit_price,
            created_at=timestamp,
            active_after_bar=self._bar_index + self.latency_bars,
        )
        self.open_orders[order.order_id] = order
        self._enqueue(
            EventType.ORDER,
            timestamp,
            {"order_id": order.order_id, "action": "ACCEPTED"},
        )
        return order.order_id

    def _process_order(self, payload: dict) -> None:
        # Order acceptance is bookkeeping only; matching occurs on later MARKET events.
        return

    def _match_open_orders(self, bar: BarSnapshot) -> None:
        filled_ids: List[int] = []
        for order_id, order in self.open_orders.items():
            if bar.bar_index < order.active_after_bar:
                continue
            if order.symbol != bar.symbol:
                continue

            filled = False
            fill_price = order.limit_price

            if order.side == "BUY" and bar.low <= order.limit_price:
                # Resting bid lifted when trade prints at or below our bid.
                filled = True
            elif order.side == "SELL" and bar.high >= order.limit_price:
                filled = True

            if not filled:
                continue

            cost = fill_price * order.quantity
            if order.side == "BUY":
                if cost > self.cash:
                    continue
                self.cash -= cost
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.quantity
            else:
                held = self.positions.get(order.symbol, 0.0)
                if order.quantity > held:
                    continue
                self.cash += cost
                self.positions[order.symbol] = held - order.quantity

            self.fills.append(
                FillRecord(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=fill_price,
                    timestamp=bar.timestamp,
                    bar_index=bar.bar_index,
                )
            )
            self._enqueue(
                EventType.FILL,
                bar.timestamp,
                {"order_id": order_id, "price": fill_price, "quantity": order.quantity},
            )
            filled_ids.append(order_id)

        for order_id in filled_ids:
            del self.open_orders[order_id]

    def _mark_equity(self, timestamp: pd.Timestamp) -> None:
        if self._current_bar is None:
            return
        mark = self._current_bar.close
        holdings = sum(
            qty * mark for sym, qty in self.positions.items() if sym == self._current_bar.symbol
        )
        self.equity_curve.append((timestamp, self.cash + holdings))

    def run(self, symbol: str = "ASSET") -> pd.DataFrame:
        """Replay all bars through the event queue."""
        self._event_queue.clear()
        self.cash = self.initial_cash
        self.positions.clear()
        self.open_orders.clear()
        self.fills.clear()
        self.equity_curve.clear()
        self._history_rows.clear()
        self._bar_index = -1

        for timestamp, row in self._bars.iterrows():
            self._bar_index += 1
            bar = self._bar_snapshot(row, pd.Timestamp(timestamp), symbol)
            self._enqueue(EventType.MARKET, bar.timestamp, {"bar": bar})

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.event_type == EventType.MARKET:
                self._process_market(event.payload)
            elif event.event_type == EventType.SIGNAL:
                self._process_signal(event.payload)
            elif event.event_type == EventType.ORDER:
                self._process_order(event.payload)
            elif event.event_type == EventType.FILL:
                continue

        if not self.equity_curve:
            return pd.DataFrame(columns=["equity"])
        idx, values = zip(*self.equity_curve)
        return pd.DataFrame({"equity": values}, index=pd.DatetimeIndex(idx))


# ---------------------------------------------------------------------------
# Look-ahead bias audit
# ---------------------------------------------------------------------------


@dataclass
class BiasAuditResult:
    passed: bool
    violations: List[str]


class LookAheadBiasAudit:
    """
    Automated checks for common leakage patterns.

    Tests
    -----
    1. Un-shifted rolling features vs PiT-shifted features produce different signals.
    2. Strategy history slice never includes future rows relative to current bar.
    3. Fills never occur before order latency elapses.
    4. Unbounded forward-fill is flagged when gap exceeds max_ffill_bars.
    5. Oracle signal (using t+1 close) is detected as leakage.
    """

    def __init__(self, max_ffill_bars: int = 0) -> None:
        self.max_ffill_bars = max_ffill_bars

    def audit_feature_shift(
        self,
        raw: pd.Series,
        pit: pd.Series,
    ) -> BiasAuditResult:
        violations: List[str] = []
        if len(raw) < 5:
            return BiasAuditResult(True, violations)

        pit_clean = pit.dropna()
        if pit_clean.empty:
            return BiasAuditResult(True, violations)

        unshifted_ma = raw.rolling(5).mean()
        shifted_ma = unshifted_ma.shift(1)
        future_close = raw.shift(-1)

        overlap = pit_clean.index.intersection(unshifted_ma.dropna().index)
        if overlap.size:
            if np.allclose(
                pit_clean.loc[overlap].values,
                unshifted_ma.loc[overlap].values,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            ):
                violations.append(
                    "PiT feature matches unshifted rolling mean (missing .shift(1))."
                )

        future_overlap = pit_clean.index.intersection(future_close.dropna().index)
        if future_overlap.size:
            if np.allclose(
                pit_clean.loc[future_overlap].values,
                future_close.loc[future_overlap].values,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            ):
                violations.append("PiT feature matches close.shift(-1) (future leakage).")

        contemporaneous = pit_clean.index.intersection(raw.dropna().index)
        if contemporaneous.size:
            if np.allclose(
                pit_clean.loc[contemporaneous].values,
                raw.loc[contemporaneous].values,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            ):
                violations.append("PiT feature matches contemporaneous raw close.")

        expected = shifted_ma.dropna()
        expected_overlap = pit_clean.index.intersection(expected.index)
        if expected_overlap.size:
            matches_shifted = np.allclose(
                pit_clean.loc[expected_overlap].values,
                expected.loc[expected_overlap].values,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )
            if not matches_shifted and not violations:
                violations.append(
                    "PiT feature does not align with rolling(5).mean().shift(1) reference."
                )

        return BiasAuditResult(len(violations) == 0, violations)

    def audit_bounded_ffill(self, series: pd.Series) -> BiasAuditResult:
        violations: List[str] = []
        if series.isna().sum() == 0:
            return BiasAuditResult(True, violations)

        filled = series.ffill()
        run = 0
        max_run = 0
        for is_nan, is_filled in zip(series.isna(), filled.notna() & series.isna()):
            if is_nan and is_filled:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        if max_run > self.max_ffill_bars:
            violations.append(
                f"Forward-fill run length {max_run} exceeds max_ffill_bars={self.max_ffill_bars}."
            )
        return BiasAuditResult(len(violations) == 0, violations)

    def audit_detects_future_feature(self, raw: pd.Series) -> BiasAuditResult:
        """A shift(-1) 'feature' must be flagged as non-PiT."""
        violations: List[str] = []
        leaky = raw.shift(-1)
        result = self.audit_feature_shift(raw, leaky)
        if result.passed:
            violations.append(
                "Failed to flag shift(-1) feature (future close) as look-ahead biased."
            )
        return BiasAuditResult(len(violations) == 0, violations)

    def audit_history_slice(self, bars: pd.DataFrame) -> BiasAuditResult:
        violations: List[str] = []
        seen_future: List[pd.Timestamp] = []

        class SliceGuard:
            def on_bar(self, history: pd.DataFrame, bar: BarSnapshot) -> List[dict]:
                if not history.empty:
                    latest = history.index.max()
                    if latest > bar.timestamp:
                        seen_future.append(latest)
                return []

        tester = EventDrivenBacktester(bars, SliceGuard().on_bar)
        tester.run()
        if seen_future:
            violations.append(
                f"Strategy received {len(seen_future)} history slices with future timestamps."
            )
        return BiasAuditResult(len(violations) == 0, violations)

    def audit_fill_latency(
        self,
        bars: pd.DataFrame,
        latency_bars: int = 1,
    ) -> BiasAuditResult:
        violations: List[str] = []

        class AlwaysBuy:
            def on_bar(self, history: pd.DataFrame, bar: BarSnapshot) -> List[dict]:
                return [{"side": "BUY", "quantity": 1.0}]

        tester = EventDrivenBacktester(bars, AlwaysBuy().on_bar, latency_bars=latency_bars)
        bar_at_submit: Dict[int, int] = {}
        original_submit = tester.submit_limit_order

        def tracked_submit(*args, **kwargs):
            oid = original_submit(*args, **kwargs)
            bar_at_submit[oid] = tester._bar_index
            return oid

        tester.submit_limit_order = tracked_submit  # type: ignore[method-assign]
        tester.run()

        for fill in tester.fills:
            placed = bar_at_submit.get(fill.order_id, -1)
            if fill.bar_index < placed + latency_bars:
                violations.append(
                    f"Fill on bar {fill.bar_index} before latency elapsed "
                    f"(order placed bar {placed}, latency={latency_bars})."
                )

        return BiasAuditResult(len(violations) == 0, violations)

    def run_all(self, bars: pd.DataFrame, pit_feature: pd.Series) -> BiasAuditResult:
        raw = bars["close"]
        results = [
            self.audit_feature_shift(raw, pit_feature),
            self.audit_bounded_ffill(pit_feature),
            self.audit_history_slice(bars),
            self.audit_fill_latency(bars, latency_bars=1),
            self.audit_detects_future_feature(raw),
        ]
        violations: List[str] = []
        for result in results:
            violations.extend(result.violations)
        return BiasAuditResult(len(violations) == 0, violations)


def run_lookahead_bias_audit(
    n_bars: int = 120,
    *,
    seed: int = 42,
) -> BiasAuditResult:
    """
    Executable audit harness for CI / manual review.

    Uses synthetic OHLCV with a correctly shifted PiT feature.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_bars)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n_bars))
    bars = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n_bars),
            "high": close + rng.uniform(0.05, 0.5, n_bars),
            "low": close - rng.uniform(0.05, 0.5, n_bars),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_bars),
        },
        index=idx,
    )
    pit = bars["close"].rolling(5).mean().shift(1)
    audit = LookAheadBiasAudit(max_ffill_bars=0)
    return audit.run_all(bars, pit)


if __name__ == "__main__":
    outcome = run_lookahead_bias_audit()
    status = "PASSED" if outcome.passed else "FAILED"
    print(f"Look-ahead bias audit: {status}")
    for msg in outcome.violations:
        print(f"  - {msg}")
    raise SystemExit(0 if outcome.passed else 1)
