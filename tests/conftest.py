"""Shared test fixtures."""

import numpy as np
import pandas as pd

from data.harvester.storage import BarStore


def seed_market_db(db_path) -> None:
    store = BarStore(db_path)
    for sym in ["NVDA", "AMD", "SPY"]:
        daily = pd.date_range("2023-01-01", periods=300, freq="D")
        n = len(daily)
        rng = np.random.default_rng(hash(sym) % 2**32)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 2e6),
            },
            index=daily,
        )
        store.upsert_bars(sym, "1d", bars)

    intra = pd.date_range("2024-01-02 09:30", periods=78, freq="15min")
    n = len(intra)
    rng = np.random.default_rng(99)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 5e5),
        },
        index=intra,
    )
    store.upsert_bars("NVDA", "15m", bars)
    store.upsert_bars("NVDA", "5m", bars)
