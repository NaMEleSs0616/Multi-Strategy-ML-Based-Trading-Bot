"""
Alpaca passive LIMIT execution (stub when API keys absent).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.interfaces import (
    AbstractExecutionHandler,
    LimitOrderRequest,
    OrderAck,
    OrderSide,
    OrderStatus,
    OrderType,
    PassiveQuote,
)
from execution.limit_orders import limit_price_for_side, quote_from_close

logger = logging.getLogger(__name__)


def _keys_present() -> bool:
    return bool(os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    )


class AlpacaExecutionHandler(AbstractExecutionHandler):
    """Passive limit orders only; no-op stub without credentials."""

    def __init__(self, *, paper: bool = True) -> None:
        self._paper = paper
        self._available = _keys_present()
        self._positions: dict[str, float] = {}
        self._orders: dict[str, OrderAck] = {}
        self._equity = 100_000.0
        self._quotes: dict[str, PassiveQuote] = {}
        self._client = None

        if self._available:
            try:
                from alpaca.trading.client import TradingClient

                api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
                secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
                self._client = TradingClient(api_key, secret, paper=paper)
            except Exception as exc:
                logger.warning("Alpaca trading client init failed: %s", exc)
                self._available = False
        else:
            logger.info("Alpaca keys missing — execution handler runs in stub mode")

    def get_quote(self, symbol: str) -> PassiveQuote:
        if symbol in self._quotes:
            q = self._quotes[symbol]
            return q
        return PassiveQuote(symbol=symbol, bid=100.0, ask=100.05, timestamp=datetime.now(timezone.utc))

    def set_quote_from_close(self, symbol: str, close: float, spread_bps: float = 5.0) -> PassiveQuote:
        q = quote_from_close(close, spread_bps)
        pq = PassiveQuote(symbol=symbol, bid=q.bid, ask=q.ask, timestamp=datetime.now(timezone.utc))
        self._quotes[symbol] = pq
        return pq

    def submit_limit_order(self, request: LimitOrderRequest) -> OrderAck:
        self.validate_limit_only(OrderType.LIMIT)
        order_id = request.client_order_id or str(uuid.uuid4())

        if not self._available or self._client is None:
            ack = OrderAck(
                order_id=order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                limit_price=request.limit_price,
                status=OrderStatus.ACCEPTED,
                raw={"stub": True},
            )
            self._orders[order_id] = ack
            return ack

        try:
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
            from alpaca.trading.requests import LimitOrderRequest as AlpacaLimit

            side = AlpacaSide.BUY if request.side == OrderSide.BUY else AlpacaSide.SELL
            alpaca_req = AlpacaLimit(
                symbol=request.symbol,
                qty=request.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=request.limit_price,
                client_order_id=order_id,
            )
            resp = self._client.submit_order(alpaca_req)
            ack = OrderAck(
                order_id=str(resp.id),
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                limit_price=request.limit_price,
                status=OrderStatus.ACCEPTED,
                raw={"alpaca_id": str(resp.id)},
            )
            self._orders[ack.order_id] = ack
            return ack
        except Exception as exc:
            logger.error("Alpaca order failed: %s", exc)
            return OrderAck(
                order_id=order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                limit_price=request.limit_price,
                status=OrderStatus.REJECTED,
                raw={"error": str(exc)},
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self._available or self._client is None:
            ack = self._orders.get(order_id)
            if ack:
                ack.status = OrderStatus.CANCELLED
                return True
            return False
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderAck]:
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return [o for o in orders if o.status in {OrderStatus.ACCEPTED, OrderStatus.PENDING}]

    def get_positions(self) -> dict[str, float]:
        if self._available and self._client is not None:
            try:
                positions = self._client.get_all_positions()
                return {p.symbol: float(p.qty) for p in positions}
            except Exception:
                pass
        return dict(self._positions)

    def get_equity(self) -> float:
        if self._available and self._client is not None:
            try:
                account = self._client.get_account()
                return float(account.equity)
            except Exception:
                pass
        return self._equity
