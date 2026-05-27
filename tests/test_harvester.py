"""SQLite bar store tests."""

import numpy as np
import pandas as pd

from data.harvester.storage import BarStore


def test_upsert_and_load(tmp_path):
    store = BarStore(tmp_path / "test.db")
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    n = 10
    close = 100 + np.arange(n)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    assert store.upsert_bars("NVDA", "1d", bars) == 10
    loaded = store.load_bars("NVDA", "1d")
    assert len(loaded) == 10
    assert float(loaded.iloc[-1]["close"]) == float(close[-1])
