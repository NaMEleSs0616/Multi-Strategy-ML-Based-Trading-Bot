"""
Event-driven SPY execution layer for capital routing.

Rebalances a single liquid ETF via passive limit orders when target exposure changes.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from backtesting.event_driven_backtester import BarSnapshot


class RoutingExecutionStrategy:
    """
    Maps portfolio target exposure into SPY limit orders.

    ``exposure_fn(bar_index)`` returns desired long exposure in [0, max_exposure]
    (fraction of equity to allocate to SPY).
    """

    def __init__(
        self,
        exposure_fn: Callable[[int], float],
        *,
        max_exposure: float = 1.0,
        rebalance_threshold: float = 0.05,
        trade_lot_value: float = 25_000.0,
    ) -> None:
        self.exposure_fn = exposure_fn
        self.max_exposure = max_exposure
        self.rebalance_threshold = rebalance_threshold
        self.trade_lot_value = trade_lot_value
        self._last_exposure = 0.0
        self._equity_estimate = trade_lot_value * 4

    def on_bar(self, history: pd.DataFrame, bar: BarSnapshot) -> List[dict]:
        target = float(np.clip(self.exposure_fn(bar.bar_index), 0.0, self.max_exposure))
        delta = target - self._last_exposure
        if abs(delta) < self.rebalance_threshold:
            return []

        self._last_exposure = target
        notional = abs(delta) * self._equity_estimate
        qty = max(1.0, notional / max(bar.close, 1e-6))

        if delta > 0:
            return [{"side": "BUY", "quantity": qty}]
        return [{"side": "SELL", "quantity": qty}]
