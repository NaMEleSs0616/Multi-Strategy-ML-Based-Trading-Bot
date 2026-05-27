"""Macro and sentiment feature fallbacks."""

import os

import numpy as np
import pandas as pd

from data.features.macro_fred import fetch_macro_frame
from data.features.sentiment_edgar import daily_sentiment_series


def test_macro_fallback_without_fred_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    macro = fetch_macro_frame(idx, pit_shift=1)
    assert list(macro.columns) == ["yield_spread_10y2y", "vix_level"]
    assert len(macro) == 30
    assert macro.isna().sum().sum() == 0


def test_sentiment_skip_edgar(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_EDGAR_SENTIMENT", "1")
    idx = pd.date_range("2024-06-01", periods=20, freq="D")
    series = daily_sentiment_series("NVDA", idx, pit_shift=1)
    assert series.name == "finbert_sentiment"
    assert series.between(-1, 1).all()
    assert series.isna().sum() == 0
