"""
Variance Risk Premium (VRP) fade / harvesting.

Logic
-----
Treat VRP as the compensation for being short volatility. When implied
volatility (VIX) is far above realized volatility, fear is extreme and
forward equity returns tend to be positive as volatility mean-reverts.

We compute:
  - IV  = VIX level / 100  (annualized implied vol, decimal)
  - RV  = rolling std(SPY daily returns, window=30) * sqrt(252) (annualized realized vol)
  - VRP = IV - RV

Signal
------
    position_t = +1 if zscore(VRP)_{t-1} > z_threshold else 0

All state is PiT-shifted before generating the position; strategy returns
are ``position_{t-1} * SPY_return_t``.

Graceful fallback
-----------------
If VIX is unavailable (no FRED key) this leg emits zeros and becomes inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import DailyStrategy, DailyStrategyConfig, pit_shift, position_returns
from strategies.vix_fade import fetch_vix_series


@dataclass(frozen=True)
class VRPHarvestConfig(DailyStrategyConfig):
    rv_window: int = 30
    z_window: int = 252
    z_threshold: float = 2.0
    annualization: float = 252.0


class VarianceRiskPremiumFade(DailyStrategy):
    def __init__(
        self,
        spy_returns: pd.Series,
        index: pd.DatetimeIndex,
        *,
        config: Optional[VRPHarvestConfig] = None,
    ) -> None:
        super().__init__(config or VRPHarvestConfig())
        self._spy_returns = spy_returns.astype(float)
        self._index = pd.DatetimeIndex(index).normalize()

    @property
    def cfg(self) -> VRPHarvestConfig:  # type: ignore[override]
        return self.config  # type: ignore[return-value]

    def returns(self) -> pd.Series:
        idx = self._index
        spy_r = (
            pd.Series(self._spy_returns.values, index=pd.DatetimeIndex(self._spy_returns.index))
            .astype(float)
            .reindex(idx)
            .fillna(0.0)
        )

        # VIX level (annualized implied vol, percent). Degrades to zeros if FRED absent.
        vix_level = fetch_vix_series(idx, pit_shift_bars=0).astype(float).reindex(idx).fillna(0.0)
        if float(vix_level.abs().sum()) <= 0.0:
            return pd.Series(0.0, index=idx)

        iv = (vix_level / 100.0).clip(lower=0.0)

        rv = spy_r.rolling(self.cfg.rv_window, min_periods=self.cfg.rv_window).std(ddof=0)
        rv = rv * float(np.sqrt(self.cfg.annualization))
        rv = rv.fillna(0.0).clip(lower=0.0)

        vrp = (iv - rv).fillna(0.0)

        mu = vrp.rolling(self.cfg.z_window, min_periods=max(30, self.cfg.z_window // 4)).mean()
        sig = vrp.rolling(self.cfg.z_window, min_periods=max(30, self.cfg.z_window // 4)).std(ddof=0)
        sig = sig.replace(0, np.nan)
        z = (vrp - mu) / sig
        z = z.fillna(0.0)

        z_pit = pit_shift(z, self.cfg.pit_shift_bars)
        position = pd.Series(0.0, index=idx)
        position[z_pit > float(self.cfg.z_threshold)] = 1.0

        return position_returns(position, spy_r, pit_shift_bars=1).fillna(0.0)


def vrp_harvesting_returns(
    spy_returns: pd.Series,
    index: pd.DatetimeIndex,
    *,
    rv_window: int = 30,
    z_window: int = 252,
    z_threshold: float = 2.0,
    pit_shift_bars: int = 1,
) -> pd.Series:
    """Functional wrapper matching the rest of the strategy bank style."""
    strat = VarianceRiskPremiumFade(
        spy_returns,
        index,
        config=VRPHarvestConfig(
            rv_window=rv_window,
            z_window=z_window,
            z_threshold=z_threshold,
            pit_shift_bars=pit_shift_bars,
        ),
    )
    return strat.returns()

