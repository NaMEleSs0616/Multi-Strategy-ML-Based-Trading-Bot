"""
Execution handler for event-driven backtests (passive limits, simulated fills).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from core.interfaces import (
    AbstractExecutionHandler,
    FillEvent,
    LimitOrderRequest,
    OrderAck,
    OrderSide,
    OrderStatus,
    OrderType,
    PassiveQuote,
)
from execution.limit_orders import limit_price_for_side, quote_from_close


class BacktestExecutionHandler(AbstractExecutionHandler):
    """
    Simulated limit-book for bar-by-bar backtests.

    Call :meth:`on_bar` each market event; fills resting limits when price trades through.
    """

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        spread_bps: float = 5.0,
        on_fill: Optional[Callable[[FillEvent], None]] = None,
    ) -> None:
        self._cash = initial_cash
        self._spread_bps = spread_bps
        self._positions: dict[str, float] = {}
        self._open_orders: dict[str, OrderAck] = {}
        self._pending: list[tuple[OrderAck, LimitOrderRequest]] = []
        self._on_fill = on_fill
        self._last_close: dict[str, float] = {}
        self._bar_index = 0

    def get_quote(self, symbol: str) -> PassiveQuote:
        close = self._last_close.get(symbol.upper(), 100.0)
        q = quote_from_close(close, self._spread_bps)
        return PassiveQuote(symbol=symbol.upper(), bid=q.bid, ask=q.ask)

    def submit_limit_order(self, request: LimitOrderRequest) -> OrderAck:
        self.validate_limit_only(OrderType.LIMIT)
        order_id = request.client_order_id or str(uuid.uuid4())
        ack = OrderAck(
            order_id=order_id,
            symbol=request.symbol.upper(),
            side=request.side,
            quantity=request.quantity,
            limit_price=request.limit_price,
            status=OrderStatus.ACCEPTED,
        )
        self._open_orders[order_id] = ack
        self._pending.append((ack, request))
        return ack

    def cancel_order(self, order_id: str) -> bool:
        ack = self._open_orders.get(order_id)
        if ack is None:
            return False
        ack.status = OrderStatus.CANCELLED
        self._pending = [(a, r) for a, r in self._pending if a.order_id != order_id]
        return True

    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderAck]:
        orders = [o for o in self._open_orders.values() if o.status == OrderStatus.ACCEPTED]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_equity(self) -> float:
        equity = self._cash
        for sym, qty in self._positions.items():
            px = self._last_close.get(sym, 0.0)
            equity += qty * px
        return equity

    def on_bar(
        self,
        symbol: str,
        *,
        open_: float,
        high: float,
        low: float,
        close: float,
        timestamp: Optional[datetime] = None,
    ) -> list[FillEvent]:
        """Match resting limits against the bar range; update positions."""
        sym = symbol.upper()
        self._last_close[sym] = close
        self._bar_index += 1
        ts = timestamp or datetime.now(timezone.utc)
        fills: list[FillEvent] = []

        remaining: list[tuple[OrderAck, LimitOrderRequest]] = []
        for ack, req in self._pending:
            if ack.status != OrderStatus.ACCEPTED:
                continue
            filled = False
            if req.side == OrderSide.BUY and low <= req.limit_price:
                px = min(req.limit_price, close)
                filled = True
            elif req.side == OrderSide.SELL and high >= req.limit_price:
                px = max(req.limit_price, close)
                filled = True

            if filled:
                ack.status = OrderStatus.FILLED
                qty = req.quantity
                cost = qty * px
                if req.side == OrderSide.BUY:
                    self._cash -= cost
                    self._positions[sym] = self._positions.get(sym, 0.0) + qty
                else:
                    self._cash += cost
                    self._positions[sym] = self._positions.get(sym, 0.0) - qty
                fill = FillEvent(
                    order_id=ack.order_id,
                    symbol=sym,
                    side=req.side,
                    quantity=qty,
                    price=px,
                    timestamp=ts,
                )
                fills.append(fill)
                if self._on_fill:
                    self._on_fill(fill)
            else:
                remaining.append((ack, req))

        self._pending = remaining
        return fills

    def submit_passive_from_signal(
        self,
        symbol: str,
        side: str,
        quantity: float,
        close: float,
    ) -> OrderAck:
        """Convenience: price limit at bid (buy) or ask (sell)."""
        quote = quote_from_close(close, self._spread_bps)
        if side.upper() == "BUY":
            limit_px = limit_price_for_side("BUY", quote)
            order_side = OrderSide.BUY
        else:
            limit_px = limit_price_for_side("SELL", quote)
            order_side = OrderSide.SELL
        return self.submit_limit_order(
            LimitOrderRequest(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                limit_price=limit_px,
            )
        )
