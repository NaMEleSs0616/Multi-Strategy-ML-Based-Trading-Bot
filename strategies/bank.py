"""
Strategy bank: align stat-arb, vol breakout, and mean reversion to a master timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings_store import PROJECT_ROOT, load_settings
from adapters.factory import create_data_handler
from core.interfaces import BarQuery
from data.harvester.storage import BarStore
from rl.gym_trading_env import StrategyReturns
from strategies.base import align_returns, compound_to_daily
from strategies.mean_reversion import mean_reversion_returns
from strategies.stat_arb import select_best_pair, select_pair, stat_arb_returns
from strategies.vol_breakout import vol_breakout_returns


@dataclass
class MarketBundle:
    """Aligned market data for RL training."""

    master_index: pd.DatetimeIndex
    features_index: pd.DatetimeIndex
    strategy_returns: StrategyReturns
    spy_returns: np.ndarray
    primary_bars: pd.DataFrame


def _load_bars(
    store: BarStore,
    symbol: str,
    interval: str,
    settings: dict[str, Any],
    data_handler: Optional[Any] = None,
) -> pd.DataFrame:
    bars = store.load_bars(symbol, interval)
    if len(bars) >= 60:
        return bars

    handler = data_handler or create_data_handler(settings)
    period = settings.get("data", {}).get("yfinance_period", "2y")
    intraday_period = settings.get("data", {}).get("intraday_period", "60d")
    fetch_period = period if interval in {"1d", "1wk"} else intraday_period
    fetched = handler.fetch_bars(
        BarQuery(symbol=symbol, interval=interval, period=fetch_period)
    )
    if not fetched.empty:
        store.upsert_bars(symbol, interval, fetched)
    return fetched


def _daily_close_series(bars: pd.DataFrame) -> pd.Series:
    close = bars["close"].astype(float)
    idx = pd.to_datetime(bars.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame = pd.DataFrame({"close": close}, index=idx)
    daily = frame.groupby(pd.Grouper(freq="D"))["close"].last().dropna()
    daily.index = pd.to_datetime(daily.index).normalize()
    return daily


def build_strategy_bank(
    settings: Optional[dict[str, Any]] = None,
    *,
    primary_symbol: Optional[str] = None,
) -> MarketBundle:
    settings = settings or load_settings()
    strat_cfg = settings.setdefault(
        "strategies",
        {
            "pair": ["NVDA", "AMD"],
            "stat_arb_z_window": 20,
            "stat_arb_entry_z": 2.0,
            "vol_atr_window": 14,
            "vol_atr_mult": 1.5,
            "vol_lookback": 20,
            "mr_rsi_window": 14,
            "mr_bb_window": 20,
        },
    )
    feat_cfg = settings["features"]
    pit_shift = int(feat_cfg.get("pit_shift", 1))
    data_cfg = settings["data"]

    db_path = PROJECT_ROOT / data_cfg["sqlite_path"]
    store = BarStore(db_path)

    primary = primary_symbol or settings["universe"]["equities"][0]
    daily_iv = data_cfg.get("daily_interval", "1d")
    iv_15m = "15m"
    iv_5m = "5m"
    for iv in data_cfg.get("intraday_intervals", ["5m", "15m"]):
        if iv == "15m":
            iv_15m = iv
        if iv == "5m":
            iv_5m = iv

    primary_daily = _load_bars(store, primary, daily_iv, settings)
    if primary_daily.empty:
        raise RuntimeError(f"No daily bars for {primary}. Run: python scripts/sync_data.py")

    master_index = pd.to_datetime(primary_daily.index).normalize()
    if master_index.tz is not None:
        master_index = master_index.tz_localize(None)

    # --- Stat arb (daily) ---
    pair = tuple(str(s) for s in strat_cfg.get("pair", ["NVDA", "AMD"]))
    cost_bps = float(strat_cfg.get("transaction_cost_bps", 0.0))
    dynamic_pair = bool(strat_cfg.get("dynamic_pair_selection", False))

    universe_syms = [str(s) for s in settings.get("universe", {}).get("equities", list(pair))]
    load_syms = list(dict.fromkeys(universe_syms if dynamic_pair else list(pair)))

    prices: dict[str, pd.Series] = {}
    for sym in load_syms:
        bars = _load_bars(store, sym, daily_iv, settings)
        if not bars.empty:
            prices[sym] = _daily_close_series(bars)

    if dynamic_pair and len(prices) >= 2:
        leg_a, leg_b, beta, _pval = select_best_pair(prices, load_syms)
    else:
        leg_a, leg_b, beta = select_pair(prices, pair)  # type: ignore[arg-type]

    if leg_a not in prices or leg_b not in prices:
        raise KeyError(f"Missing prices for stat-arb pair ({leg_a}, {leg_b})")

    stat = stat_arb_returns(
        prices[leg_a],
        prices[leg_b],
        hedge_ratio=beta,
        z_window=int(strat_cfg.get("stat_arb_z_window", 20)),
        entry_z=float(strat_cfg.get("stat_arb_entry_z", 2.0)),
        pit_shift_bars=pit_shift,
        transaction_cost_bps=cost_bps,
    )
    stat_daily = align_returns(stat, master_index)

    # --- Vol breakout (15m -> daily) ---
    bars_15m = _load_bars(store, primary, iv_15m, settings)
    if bars_15m.empty:
        vol_daily = np.zeros(len(master_index))
    else:
        vol = vol_breakout_returns(
            bars_15m,
            atr_window=int(strat_cfg.get("vol_atr_window", feat_cfg.get("atr_window", 14))),
            atr_mult=float(strat_cfg.get("vol_atr_mult", 1.5)),
            lookback=int(strat_cfg.get("vol_lookback", 20)),
            pit_shift_bars=pit_shift,
        )
        vol_daily = compound_to_daily(vol, master_index).to_numpy(dtype=np.float64)

    # --- Mean reversion (5m -> daily) ---
    bars_5m = _load_bars(store, primary, iv_5m, settings)
    if bars_5m.empty:
        mr_daily = np.zeros(len(master_index))
    else:
        mr = mean_reversion_returns(
            bars_5m,
            rsi_window=int(strat_cfg.get("mr_rsi_window", 14)),
            pit_shift_bars=pit_shift,
            bb_window=int(strat_cfg.get("mr_bb_window", 20)),
        )
        mr_daily = compound_to_daily(mr, master_index).to_numpy(dtype=np.float64)

    # --- SPY benchmark ---
    spy_bars = _load_bars(store, "SPY", daily_iv, settings)
    if spy_bars.empty:
        spy_daily = primary_daily["close"].pct_change().fillna(0.0)
    else:
        spy_daily = _daily_close_series(spy_bars).pct_change().fillna(0.0)
    spy_aligned = align_returns(spy_daily, master_index)

    n = len(master_index)
    if not (len(stat_daily) == len(vol_daily) == len(mr_daily) == n):
        raise ValueError("Strategy return lengths do not match master timeline")

    return MarketBundle(
        master_index=master_index,
        features_index=master_index,
        strategy_returns=StrategyReturns(
            stat_arb=stat_daily,
            vol_breakout=vol_daily,
            mean_reversion=mr_daily,
        ),
        spy_returns=spy_aligned,
        primary_bars=primary_daily,
    )
