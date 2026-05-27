"""
AFML-style fractional differentiation (López de Prado, ch. 5).

Weights: w_0 = 1; w_k = -w_{k-1} * (d - k + 1) / k
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fracdiff_weights(
    d: float,
    *,
    threshold: float = 1e-3,
    max_size: int = 2000,
) -> np.ndarray:
    """
    Binomial-expansion weights for fractional differentiation order ``d``.

    Stops when |w_k| < ``threshold`` or ``max_size`` weights are collected.
    """
    if not 0.0 <= d <= 1.0:
        raise ValueError(f"d must be in [0, 1], got {d}")

    weights = [1.0]
    k = 1
    while k < max_size:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.asarray(weights, dtype=np.float64)


def fractional_diff(
    series: pd.Series,
    d: float,
    *,
    threshold: float = 1e-3,
) -> pd.Series:
    """
    Apply fractional differentiation to ``series`` (typically log prices).

    Uses the minimum window length implied by ``threshold`` on weights.
    """
    values = series.astype(float).to_numpy()
    n = len(values)
    if n == 0:
        return pd.Series(dtype=float, index=series.index)

    weights = fracdiff_weights(d, threshold=threshold)
    width = min(len(weights), n)
    if width < 1:
        return pd.Series(np.nan, index=series.index)
    use_w = weights[:width]
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(width - 1, n):
        window = values[i - width + 1 : i + 1][::-1]
        if np.any(np.isnan(window)):
            continue
        out[i] = float(np.dot(use_w, window))

    return pd.Series(out, index=series.index, name=series.name)
