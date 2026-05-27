"""Fractional differentiation tests."""

import numpy as np
import pandas as pd

from data.features.fracdiff import fracdiff_weights, fractional_diff
from data.features.pit_features import build_pit_feature_frame


def test_fracdiff_weights_sum_properties():
    w = fracdiff_weights(0.4, max_size=50)
    assert len(w) >= 2
    assert w[0] == 1.0
    assert np.all(np.isfinite(w))


def test_fractional_diff_produces_finite_series():
    rng = np.random.default_rng(0)
    n = 200
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)))
    log_p = np.log(close)
    fd = fractional_diff(log_p, d=0.4).dropna()
    assert len(fd) > 50
    assert np.isfinite(fd).all()


def test_pit_features_frac_diff_column():
    n = 120
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1e6, 2e6, n),
        }
    )
    pit = build_pit_feature_frame(bars, pit_shift=1, frac_diff_d=0.4)
    assert "frac_diff" in pit.columns
    assert pit.isna().sum().sum() == 0
