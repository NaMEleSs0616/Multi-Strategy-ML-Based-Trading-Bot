"""
Live training monitor — bars, PiT features, strategy returns, xLSTM/PPO metrics.

Launch:
    ./scripts/run_training_monitor.sh
    # or: PYTHONPATH=. streamlit run frontend/training_monitor.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_store import load_settings
from pipeline.training_monitor import build_monitor_snapshot

st.set_page_config(
    page_title="Training Monitor",
    page_icon="📊",
    layout="wide",
)

REFRESH_SECONDS = 10


@st.cache_data(ttl=5)
def _snapshot(symbol: str, provider: str) -> dict:
    settings = load_settings()
    snap = build_monitor_snapshot(symbol, settings=settings)
    return {
        "settings": settings,
        "bar_inventory": snap.bar_inventory,
        "price_series": snap.price_series,
        "features": snap.features,
        "strategy_returns": snap.strategy_returns,
        "spy_returns": snap.spy_returns,
        "xlstm_history": snap.xlstm_history,
        "ppo_summary": snap.ppo_summary,
        "embeddings": snap.embeddings,
        "checkpoint_status": snap.checkpoint_status,
    }


def _render_checkpoints(status: dict) -> None:
    rows = []
    for name, meta in status.items():
        rows.append(
            {
                "artifact": name,
                "exists": meta["exists"],
                "updated": meta["mtime"],
                "size_kb": meta["size_kb"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    settings = load_settings()
    st.title("Training monitor")
    st.caption(
        f"Data provider: `{settings['data'].get('provider', 'yfinance')}` · "
        f"SQLite: `{settings['data']['sqlite_path']}`"
    )

    with st.sidebar:
        st.subheader("Refresh")
        auto = st.checkbox("Auto-refresh", value=True)
        interval = st.slider("Interval (seconds)", 5, 60, REFRESH_SECONDS, 5)
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        symbols = settings["universe"]["equities"]
        symbol = st.selectbox("Primary symbol", symbols, index=0)

    try:
        data = _snapshot(symbol, settings["data"].get("provider", "yfinance"))
    except Exception as exc:
        st.error(f"Failed to load monitor snapshot: {exc}")
        st.info("Run `python scripts/sync_data.py` first, then retry.")
        return

    inv = data["bar_inventory"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bar series", len(inv) if not inv.empty else 0)
    c2.metric("Feature rows", len(data["features"]))
    c3.metric("Strategy timeline", len(data["strategy_returns"]))
    hist = data["xlstm_history"]
    c4.metric("xLSTM epochs logged", len(hist) if hist is not None else 0)

    tab_data, tab_features, tab_strategies, tab_train, tab_ckpt = st.tabs(
        ["Market data", "PiT features", "Strategy bank", "Training curves", "Checkpoints"]
    )

    with tab_data:
        st.subheader("SQLite bar inventory")
        if inv.empty:
            st.warning("No bars in SQLite. Run `python scripts/sync_data.py`.")
        else:
            st.dataframe(inv, use_container_width=True, hide_index=True)
            st.bar_chart(inv.pivot(index="symbol", columns="interval", values="rows"))

        st.subheader(f"{symbol} close ({settings['data'].get('daily_interval', '1d')})")
        prices = data["price_series"]
        if prices.empty:
            st.warning(f"No daily bars for {symbol}.")
        else:
            st.line_chart(prices["close"])

    with tab_features:
        st.subheader(f"PiT feature preview — {symbol}")
        feats = data["features"]
        if feats.empty:
            st.warning("Feature frame is empty.")
        else:
            st.line_chart(feats)
            st.dataframe(feats.tail(20), use_container_width=True)

        emb = data["embeddings"]
        if emb is not None:
            st.subheader("xLSTM embeddings (first 3 dims, tail)")
            st.line_chart(emb)

    with tab_strategies:
        st.subheader("Aligned strategy returns (tail)")
        strat = data["strategy_returns"]
        spy = data["spy_returns"]
        if strat.empty:
            st.warning("Strategy returns unavailable.")
        else:
            st.line_chart(strat)
            combined = strat.copy()
            combined["SPY"] = spy.reindex(combined.index).fillna(0.0)
            st.area_chart(combined.cumsum())
            st.caption("Cumulative returns (aligned timeline)")

    with tab_train:
        st.subheader("xLSTM training history")
        if hist is None or hist.empty:
            st.info("No `train_history.json` yet. Run `python scripts/train_xlstm.py`.")
        else:
            if "train_mse" in hist.columns:
                st.line_chart(hist[["train_mse", "val_mse"]])
            if "train_ou" in hist.columns:
                st.line_chart(hist[["train_ou", "val_ou"]])
            if "train_loss" in hist.columns:
                st.line_chart(hist[["train_loss", "val_loss"]])
            st.dataframe(hist, use_container_width=True)

        st.subheader("PPO walk-forward summary")
        ppo = data["ppo_summary"]
        if ppo is None or ppo.empty:
            st.info("No PPO summary yet. Run `python scripts/train_ppo.py`.")
        else:
            st.dataframe(ppo, use_container_width=True, hide_index=True)
            if "test_bars" in ppo.columns:
                st.bar_chart(ppo.set_index("fold_id")["test_bars"])

    with tab_ckpt:
        st.subheader("Checkpoint files")
        _render_checkpoints(data["checkpoint_status"])

    if auto:
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
