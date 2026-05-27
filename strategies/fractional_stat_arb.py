"""
Fractional cointegration stat-arb (long-memory spread mean reversion).

Motivation
----------
Classic pairs trading assumes the spread is I(0). This variant allows
*fractional* integration by estimating a long-memory parameter d. We only
trade when:

  - The spread exhibits mean-reverting long memory: 0 < d < 0.5
  - The spread deviates materially from its rolling mean (z-score trigger)

PiT Contract
------------
All indicators (d estimate, rolling mean/std, z) are computed on a rolling
window and PiT-shifted before generating position, so the realized return
at bar t uses position_{t-1} * spread_return_t, matching other legs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import DailyStrategy, DailyStrategyConfig, pit_shift, position_returns


def _gph_d_estimate(x: np.ndarray, *, m: int = 10, eps: float = 1e-12) -> float:
    """
    Geweke–Porter–Hudak (GPH) estimator proxy for fractional differencing d.

    Uses a log-periodogram regression on the lowest m Fourier frequencies.
    This is a lightweight, dependency-free approximation suitable for
    rolling windows. Returns d clipped to [0, 1].
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 64:
        return 0.5
    x = x - float(np.mean(x))

    # Frequencies: 2πk/n for k=1..m
    m = int(max(5, min(m, n // 4)))
    k = np.arange(1, m + 1, dtype=np.float64)
    w = 2.0 * np.pi * k / n

    # Periodogram at low freqs
    fft = np.fft.fft(x)
    I = (np.abs(fft[1 : m + 1]) ** 2) / max(n, 1)
    I = np.maximum(I, eps)

    # Regress log(I(w)) on log(4 sin^2(w/2))
    X = np.log(4.0 * (np.sin(w / 2.0) ** 2) + eps)
    Y = np.log(I)
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    denom = float(np.dot(Xc, Xc))
    if denom <= eps:
        return 0.5
    slope = float(np.dot(Xc, Yc) / denom)

    # For fractional integration: slope ≈ -d
    d = -0.5 * slope
    return float(np.clip(d, 0.0, 1.0))


@dataclass(frozen=True)
class FractionalStatArbConfig(DailyStrategyConfig):
    window: int = 252
    entry_z: float = 2.0
    exit_z: float = 0.5
    d_m: int = 10
    d_min: float = 0.0
    d_max: float = 0.5
    hedge_ratio: Optional[float] = None


class FractionalStatArb(DailyStrategy):
    def __init__(
        self,
        leg_a: pd.Series,
        leg_b: pd.Series,
        *,
        config: Optional[FractionalStatArbConfig] = None,
    ) -> None:
        super().__init__(config or FractionalStatArbConfig())
        self._a = leg_a.astype(float)
        self._b = leg_b.astype(float)

    @property
    def cfg(self) -> FractionalStatArbConfig:  # type: ignore[override]
        return self.config  # type: ignore[return-value]

    def returns(self) -> pd.Series:
        aligned = pd.concat(
            [self._a.rename("a"), self._b.rename("b")], axis=1, join="inner"
        ).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)

        # Hedge ratio: simple OLS beta proxy via covariance (consistent with existing stat_arb).
        a = aligned["a"]
        b = aligned["b"]
        if self.cfg.hedge_ratio is None:
            cov = np.cov(np.log(a), np.log(b), ddof=0)
            beta = 1.0 if cov[1, 1] < 1e-12 else float(cov[0, 1] / cov[1, 1])
        else:
            beta = float(self.cfg.hedge_ratio)

        spread = np.log(a) - beta * np.log(b)
        spread_ret = spread.diff().fillna(0.0)

        w = int(self.cfg.window)
        mu = spread.rolling(w, min_periods=max(30, w // 4)).mean()
        sigma = spread.rolling(w, min_periods=max(30, w // 4)).std().replace(0, np.nan)
        z = (spread - mu) / sigma

        # Rolling d estimate on the same window.
        d_vals = np.full(len(spread), np.nan, dtype=np.float64)
        s_arr = spread.to_numpy(dtype=np.float64)
        for i in range(len(spread)):
            start = max(0, i - w + 1)
            seg = s_arr[start : i + 1]
            if seg.size < 64:
                continue
            d_vals[i] = _gph_d_estimate(seg, m=self.cfg.d_m)
        d = pd.Series(d_vals, index=spread.index).ffill().fillna(0.5)

        # PiT shift all state before generating position.
        z_pit = pit_shift(z, self.cfg.pit_shift_bars)
        d_pit = pit_shift(d, self.cfg.pit_shift_bars)

        # Only trade when long-memory mean-reversion is present.
        trade_ok = (d_pit > float(self.cfg.d_min)) & (d_pit < float(self.cfg.d_max))

        position = pd.Series(0.0, index=spread.index)
        enter_short = trade_ok & (z_pit > float(self.cfg.entry_z))
        enter_long = trade_ok & (z_pit < -float(self.cfg.entry_z))
        exit_flat = (~trade_ok) | (z_pit.abs() < float(self.cfg.exit_z))

        # Spread position: +1 means long spread (long a / short b).
        position[enter_long] = 1.0
        position[enter_short] = -1.0
        position[exit_flat] = 0.0

        return position_returns(position, spread_ret, pit_shift_bars=1).fillna(0.0)


def fractional_stat_arb_returns(
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    window: int = 252,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    d_m: int = 10,
    d_min: float = 0.0,
    d_max: float = 0.5,
    pit_shift_bars: int = 1,
    hedge_ratio: Optional[float] = None,
) -> pd.Series:
    """Functional wrapper matching existing bank style."""
    strat = FractionalStatArb(
        leg_a,
        leg_b,
        config=FractionalStatArbConfig(
            window=window,
            entry_z=entry_z,
            exit_z=exit_z,
            d_m=d_m,
            d_min=d_min,
            d_max=d_max,
            pit_shift_bars=pit_shift_bars,
            hedge_ratio=hedge_ratio,
        ),
    )
    return strat.returns()

