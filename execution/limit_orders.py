"""
Passive limit-order helpers for live and backtest execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass
class PassiveQuote:
    bid: float
    ask: float


def quote_from_close(close: float, spread_bps: float = 5.0) -> PassiveQuote:
    half = close * (spread_bps / 10_000.0) / 2.0
    return PassiveQuote(bid=close - half, ask=close + half)


def limit_price_for_side(side: Side, quote: PassiveQuote) -> float:
    """Liquidity-providing limits: buy at bid, sell at ask."""
    if side == "BUY":
        return quote.bid
    return quote.ask


def order_notional_to_qty(notional: float, price: float, min_qty: float = 1.0) -> float:
    if price <= 0:
        raise ValueError("price must be positive")
    return max(min_qty, notional / price)
