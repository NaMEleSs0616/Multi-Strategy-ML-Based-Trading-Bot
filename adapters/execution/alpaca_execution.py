"""
Alpaca passive LIMIT execution (stub when API keys absent).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np

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


# Strategy slots ordered as they appear in StrategyReturns / paper_trading_loop:
#   0 → stat_arb             (daily, REST-poll friendly)
#   1 → vol_breakout         (15m, requires live WebSocket bars)
#   2 → mean_reversion       (5m,  requires live WebSocket bars)
#   3 → daily_momentum       (daily, REST-poll friendly)
#   4 → daily_trend          (daily, REST-poll friendly)
#   5 → vix_fade             (daily macro, REST-poll friendly)
#   6 → cross_sectional_mom  (daily, REST-poll friendly)
#
# ``STRATEGY_ORDER`` is only the *default* mapping. Callers in the live
# router (paper loop, live inference) should pass
# ``data.strategy_returns.names()`` to ``filter_weights`` so the circuit
# breaker stays correct even when operators reorder/disable individual
# legs through ``config/settings.yaml``.
STRATEGY_ORDER = (
    "stat_arb",
    "vol_breakout",
    "mean_reversion",
    "daily_momentum",
    "daily_trend",
    "vix_fade",
    "cross_sectional_mom",
    "fractional_stat_arb",
    "vrp_harvesting",
)
INTRADAY_STRATEGIES = ("vol_breakout", "mean_reversion")
DAILY_ONLY_STRATEGIES = tuple(s for s in STRATEGY_ORDER if s not in INTRADAY_STRATEGIES)


def _keys_present() -> bool:
    return bool(os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    )


class AlpacaExecutionHandler(AbstractExecutionHandler):
    """Passive limit orders only; no-op stub without credentials.

    Latency circuit breaker
    -----------------------
    When the handler is constructed in ``alpaca`` mode with ``paper=True``
    and the caller has **not** signalled an active live WebSocket feed
    (``websockets_connected=False`` — the current default), polling the
    REST API every bar at 5m / 15m would introduce unacceptable execution
    latency. To prevent that, the handler exposes ``filter_weights`` and
    ``circuit_breaker_active``: callers (paper loop, live router) must run
    PPO output through ``filter_weights`` before sizing orders. The filter
    zeros the 5m mean-reversion and 15m vol-breakout slots and renormalizes
    the surviving daily-only slots (``stat_arb`` and, when present,
    ``daily_momentum``) to preserve the original gross exposure.
    """

    def __init__(
        self,
        *,
        paper: bool = True,
        settings: Optional[dict[str, Any]] = None,
        websockets_connected: bool = False,
    ) -> None:
        self._paper = paper
        self._available = _keys_present()
        self._positions: dict[str, float] = {}
        self._orders: dict[str, OrderAck] = {}
        self._equity = 100_000.0
        self._quotes: dict[str, PassiveQuote] = {}
        self._client = None

        exec_cfg = (settings or {}).get("execution", {}) if settings else {}
        configured_mode = str(exec_cfg.get("mode", "alpaca")).lower()
        self._mode = configured_mode
        self._websockets_connected = bool(websockets_connected)

        # Circuit breaker tripped iff: alpaca live mode + paper + no live WS feed.
        # Stub-mode (no Alpaca keys) does not need protection because nothing leaves
        # the process. We still flag it for non-keyed dry runs so behavior is uniform.
        self._circuit_breaker_active: bool = (
            configured_mode == "alpaca"
            and bool(paper)
            and not self._websockets_connected
        )

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

        if self._circuit_breaker_active:
            logger.warning(
                "AlpacaExecutionHandler circuit breaker ACTIVE "
                "(mode=%s paper=%s websockets_connected=%s) — "
                "intraday strategies %s will be zeroed; capital restricted to %s.",
                configured_mode,
                paper,
                self._websockets_connected,
                INTRADAY_STRATEGIES,
                DAILY_ONLY_STRATEGIES,
            )

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def mark_websockets_connected(self, connected: bool = True) -> None:
        """Toggle the live-WS flag; once connected, the circuit breaker disarms."""
        self._websockets_connected = bool(connected)
        if connected and self._circuit_breaker_active:
            logger.info(
                "WebSocket feed active — disarming intraday latency circuit breaker"
            )
            self._circuit_breaker_active = False
        elif not connected and self._mode == "alpaca" and self._paper:
            self._circuit_breaker_active = True

    def filter_weights(
        self,
        weights: Iterable[float] | np.ndarray,
        *,
        strategy_order: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """
        Apply the latency circuit breaker to a raw weight vector.

        When tripped, the 5m mean-reversion and 15m vol-breakout slots are
        forced to zero and the surviving daily slots are renormalized so
        their gross magnitude equals the original gross exposure
        (or zeroed if it was already zero). When inactive, weights pass
        through unchanged.

        Strategy ordering: callers may pass a custom ``strategy_order`` to
        match the live ``StrategyReturns`` layout (e.g. 3-leg legacy vs.
        4-leg w/ daily_momentum); otherwise the canonical
        :data:`STRATEGY_ORDER` is sliced to match the weight length.
        """
        arr = np.asarray(list(weights), dtype=np.float64).reshape(-1)
        if not self._circuit_breaker_active:
            return arr

        if strategy_order is None:
            order = list(STRATEGY_ORDER[: arr.shape[0]])
        else:
            order = list(strategy_order)
        if len(order) != arr.shape[0]:
            raise ValueError(
                f"weights length {arr.shape[0]} != strategy_order length {len(order)}"
            )

        out = arr.copy()
        original_gross = float(np.sum(np.abs(out)))
        for i, name in enumerate(order):
            if name in INTRADAY_STRATEGIES:
                out[i] = 0.0

        surviving = float(np.sum(np.abs(out)))
        if surviving > 0 and original_gross > 0:
            out = out * (original_gross / surviving)
        return out

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
