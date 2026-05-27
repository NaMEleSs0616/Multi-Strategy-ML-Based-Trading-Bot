"""
Streamlit control panel for the multi-strategy RL trading bot.

Launch:
    PYTHONPATH=. streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_store import DEFAULT_SETTINGS_PATH, load_settings, save_settings
from rl.gym_trading_env import run_walk_forward_smoke_test
from rl.validation.purge import PurgedWalkForwardSplitter

st.set_page_config(
    page_title="RL Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]


def _parse_intervals(raw: str) -> list[str]:
    return [i.strip() for i in raw.replace("\n", ",").split(",") if i.strip()]


def init_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    if "saved_snapshot" not in st.session_state:
        st.session_state.saved_snapshot = str(st.session_state.settings)


def settings_dirty() -> bool:
    return str(st.session_state.settings) != st.session_state.saved_snapshot


def render_sidebar() -> None:
    st.sidebar.title("Control panel")
    st.sidebar.caption(f"Config: `{DEFAULT_SETTINGS_PATH.relative_to(PROJECT_ROOT)}`")

    if settings_dirty():
        st.sidebar.warning("Unsaved changes")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Save", type="primary", use_container_width=True):
        save_settings(st.session_state.settings)
        st.session_state.saved_snapshot = str(st.session_state.settings)
        st.sidebar.success("Saved")
        st.rerun()

    if col_b.button("Reload", use_container_width=True):
        st.session_state.settings = load_settings()
        st.session_state.saved_snapshot = str(st.session_state.settings)
        st.sidebar.info("Reloaded from disk")
        st.rerun()

    if st.sidebar.button("Reset defaults", use_container_width=True):
        from copy import deepcopy

        from config.settings_store import DEFAULTS

        st.session_state.settings = deepcopy(DEFAULTS)
        st.sidebar.info("Defaults loaded (not saved yet)")
        st.rerun()


def tab_universe_data(settings: dict) -> None:
    st.subheader("Universe & data")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Equities**")
        equities = st.text_area(
            "Comma-separated tickers",
            value=", ".join(settings["universe"]["equities"]),
            height=80,
            label_visibility="collapsed",
        )
        settings["universe"]["equities"] = _parse_tickers(equities)

        st.markdown("**ETFs**")
        etfs = st.text_area(
            "Comma-separated tickers",
            value=", ".join(settings["universe"]["etfs"]),
            height=80,
            label_visibility="collapsed",
        )
        settings["universe"]["etfs"] = _parse_tickers(etfs)

    with c2:
        settings["data"]["sqlite_path"] = st.text_input(
            "SQLite path",
            value=settings["data"]["sqlite_path"],
        )
        settings["data"]["daily_interval"] = st.selectbox(
            "Daily interval",
            options=["1d"],
            index=0,
        )
        intraday_raw = st.text_input(
            "Intraday intervals (comma-separated)",
            value=", ".join(settings["data"]["intraday_intervals"]),
        )
        settings["data"]["intraday_intervals"] = _parse_intervals(intraday_raw)

    st.markdown("**Data sync** (yfinance → SQLite)")
    if st.button("Sync market data", key="sync_data"):
        from data.harvester.pipeline import sync_universe

        save_settings(settings)
        with st.spinner("Downloading OHLCV…"):
            try:
                results = sync_universe(settings)
            except Exception as exc:
                st.error(str(exc))
                return
        st.success(f"Synced {len(results)} symbol/interval series")


def tab_strategies(settings: dict) -> None:
    st.subheader("Strategy bank")
    strat = settings.setdefault(
        "strategies",
        {
            "pair": ["NVDA", "AMD"],
            "stat_arb_z_window": 20,
            "stat_arb_entry_z": 2.0,
            "vol_atr_mult": 1.5,
        },
    )
    pair_raw = st.text_input(
        "Pairs leg (comma-separated)",
        value=", ".join(strat.get("pair", ["NVDA", "AMD"])),
    )
    strat["pair"] = _parse_tickers(pair_raw)
    strat["stat_arb_entry_z"] = st.slider(
        "Stat-arb entry |z|",
        1.0,
        3.5,
        float(strat.get("stat_arb_entry_z", 2.0)),
        0.1,
    )
    strat["vol_atr_mult"] = st.slider(
        "Vol breakout ATR mult",
        0.5,
        3.0,
        float(strat.get("vol_atr_mult", 1.5)),
        0.1,
    )
    strat["transaction_cost_bps"] = st.number_input(
        "Stat-arb cost (bps per leg)",
        min_value=0.0,
        max_value=50.0,
        value=float(strat.get("transaction_cost_bps", 5.0)),
        step=0.5,
    )
    strat["dynamic_pair_selection"] = st.checkbox(
        "Dynamic pair selection (universe)",
        value=bool(strat.get("dynamic_pair_selection", False)),
    )
    settings["strategies"] = strat

    if st.button("Preview strategy returns", key="preview_strategies"):
        from strategies.bank import build_strategy_bank

        save_settings(settings)
        with st.spinner("Computing PiT strategy returns…"):
            try:
                bundle = build_strategy_bank(settings)
            except Exception as exc:
                st.error(str(exc))
                return
        import pandas as pd

        df = pd.DataFrame(
            {
                "stat_arb": bundle.strategy_returns.stat_arb,
                "vol_breakout": bundle.strategy_returns.vol_breakout,
                "mean_reversion": bundle.strategy_returns.mean_reversion,
                "spy": bundle.spy_returns,
            },
            index=bundle.master_index,
        )
        st.dataframe(df.tail(20), use_container_width=True)
        st.line_chart(df[["stat_arb", "vol_breakout", "mean_reversion", "spy"]].cumsum())


def tab_features(settings: dict) -> None:
    st.subheader("PiT feature engineering")
    c1, c2 = st.columns(2)

    with c1:
        settings["features"]["frac_diff_d"] = st.slider(
            "Fractional diff (d)",
            min_value=0.0,
            max_value=1.0,
            value=float(settings["features"]["frac_diff_d"]),
            step=0.05,
            help="Differentiation order for stationarity while preserving memory.",
        )
        settings["features"]["autocorr_window"] = st.number_input(
            "Autocorrelation window",
            min_value=5,
            max_value=120,
            value=int(settings["features"]["autocorr_window"]),
            step=1,
        )

    with c2:
        settings["features"]["atr_window"] = st.number_input(
            "ATR window",
            min_value=5,
            max_value=60,
            value=int(settings["features"]["atr_window"]),
            step=1,
        )
        settings["features"]["pit_shift"] = st.number_input(
            "PiT shift (bars)",
            min_value=1,
            max_value=5,
            value=int(settings["features"]["pit_shift"]),
            step=1,
            help="Mandatory shift(1) on all rolling features to prevent look-ahead.",
        )

    feat = settings.setdefault("features", {})
    feat["include_macro"] = st.checkbox(
        "Include macro (FRED 10Y–2Y, VIX)",
        value=bool(feat.get("include_macro", False)),
        help="Requires FRED_API_KEY; zeros if unset.",
    )
    feat["include_sentiment"] = st.checkbox(
        "Include FinBERT sentiment (EDGAR)",
        value=bool(feat.get("include_sentiment", False)),
        help="Caches scores; set SKIP_EDGAR_SENTIMENT=1 to skip downloads.",
    )
    settings["features"] = feat


def tab_xlstm(settings: dict) -> None:
    st.subheader("xLSTM encoder (Stage 1)")
    xl = settings.setdefault("xlstm", {})
    c1, c2 = st.columns(2)

    with c1:
        xl["hidden_dim"] = st.number_input(
            "Hidden dim",
            min_value=32,
            max_value=512,
            value=int(xl.get("hidden_dim", 128)),
            step=32,
        )
        xl["num_blocks"] = st.number_input(
            "xLSTM blocks",
            min_value=1,
            max_value=8,
            value=int(xl.get("num_blocks", 2)),
            step=1,
        )
        xl["sequence_length"] = st.number_input(
            "Sequence length",
            min_value=10,
            max_value=252,
            value=int(xl.get("sequence_length", 60)),
            step=5,
        )
        xl["dropout"] = st.slider(
            "Dropout",
            0.0,
            0.5,
            float(xl.get("dropout", 0.1)),
            0.05,
        )

    with c2:
        settings["rl"]["embedding_dim"] = st.number_input(
            "Embedding dim (RL obs)",
            min_value=8,
            max_value=512,
            value=int(settings["rl"]["embedding_dim"]),
            step=8,
        )
        xl["learning_rate"] = st.number_input(
            "Learning rate",
            min_value=1e-5,
            max_value=1e-2,
            value=float(xl.get("learning_rate", 1e-3)),
            format="%.5f",
        )
        xl["max_epochs"] = st.number_input(
            "Max epochs",
            min_value=1,
            max_value=500,
            value=int(xl.get("max_epochs", 100)),
        )
        xl["plateau_patience"] = st.number_input(
            "Plateau patience (freeze)",
            min_value=1,
            max_value=30,
            value=int(xl.get("plateau_patience", 5)),
        )
        xl["ou_gamma"] = st.slider(
            "OU penalty γ",
            0.0,
            1.0,
            float(xl.get("ou_gamma", 0.1)),
            0.01,
            help="L_total = L_MSE + γ L_OU (Ornstein–Uhlenbeck SDE).",
        )
        xl["ou_dt"] = st.number_input(
            "OU dt",
            min_value=0.1,
            max_value=5.0,
            value=float(xl.get("ou_dt", 1.0)),
            step=0.1,
        )

    settings["xlstm"] = xl
    use_synthetic = st.checkbox("Use synthetic data", value=False, key="xlstm_synthetic")

    if st.button("Train xLSTM", type="primary", key="train_xlstm"):
        from models.xlstm.train import train_xlstm

        save_settings(settings)
        with st.spinner("Training xLSTM…"):
            try:
                result = train_xlstm(settings, use_yfinance=not use_synthetic)
            except Exception as exc:
                st.error(str(exc))
                return
        st.success(
            f"{result.epochs_run} epochs · val MSE {result.best_val_loss:.6f} · frozen={result.frozen}"
        )
        st.code(str(result.checkpoint_path))
        if result.history_path.exists():
            import json

            history = json.loads(result.history_path.read_text())
            chart = {
                "train_mse": [r.get("train_mse", r.get("train_loss", 0)) for r in history],
                "val_mse": [r.get("val_mse", r.get("val_loss", 0)) for r in history],
            }
            if history and "train_ou" in history[0]:
                chart["train_ou"] = [r["train_ou"] for r in history]
                chart["val_ou"] = [r["val_ou"] for r in history]
            st.line_chart(chart)


def tab_ppo(settings: dict) -> None:
    st.subheader("PPO router (Stage 2)")
    ppo = settings.setdefault(
        "ppo",
        {
            "total_timesteps": 20000,
            "learning_rate": 0.0003,
            "n_steps": 256,
            "batch_size": 64,
            "checkpoint_dir": "models/ppo/checkpoints",
        },
    )
    c1, c2 = st.columns(2)
    with c1:
        ppo["total_timesteps"] = st.number_input(
            "Timesteps per fold",
            min_value=1000,
            max_value=500_000,
            value=int(ppo.get("total_timesteps", 20000)),
            step=1000,
        )
        ppo["learning_rate"] = st.number_input(
            "Learning rate",
            min_value=1e-5,
            max_value=1e-2,
            value=float(ppo.get("learning_rate", 3e-4)),
            format="%.5f",
        )
    with c2:
        ppo["n_steps"] = st.number_input("Rollout n_steps", 64, 2048, int(ppo.get("n_steps", 256)))
        ppo["batch_size"] = st.number_input("Batch size", 32, 512, int(ppo.get("batch_size", 64)))
    settings["ppo"] = ppo

    require_enc = st.checkbox("Require xLSTM checkpoint", value=False)
    st.caption("Uses SQLite + strategy bank; falls back to raw features if no xLSTM checkpoint.")
    if st.button("Train PPO router", type="primary", key="train_ppo"):
        from models.ppo.train_router import train_ppo_router

        save_settings(settings)
        with st.spinner("Training PPO on walk-forward folds…"):
            try:
                result = train_ppo_router(settings, require_encoder=require_enc)
            except Exception as exc:
                st.error(str(exc))
                return
        st.success(
            f"Trained {result.folds_trained} folds on {result.n_timesteps} bars → `{result.model_path}`"
        )


def tab_rl(settings: dict) -> None:
    st.subheader("RL routing & walk-forward")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Embedding dim", int(settings["rl"]["embedding_dim"]))
        st.caption("Set on the **xLSTM** tab.")
        settings["rl"]["n_strategies"] = st.number_input(
            "Strategy count",
            min_value=2,
            max_value=10,
            value=int(settings["rl"]["n_strategies"]),
            step=1,
        )
        settings["rl"]["turnover_penalty_lambda"] = st.slider(
            "Turnover penalty λ",
            min_value=0.0,
            max_value=1.0,
            value=float(settings["rl"]["turnover_penalty_lambda"]),
            step=0.01,
            help="L1 penalty on ||w_t - w_{t-1}|| in the reward.",
        )
        risk = settings.setdefault("risk", {})
        risk["enabled"] = st.checkbox("Kelly risk overlay", value=bool(risk.get("enabled", True)))
        risk["kelly_fraction"] = st.slider(
            "Kelly fraction (×K)",
            0.0,
            1.0,
            float(risk.get("kelly_fraction", 0.5)),
            0.05,
        )
        risk["kelly_window"] = st.number_input(
            "Kelly window (days)",
            min_value=5,
            max_value=120,
            value=int(risk.get("kelly_window", 30)),
        )
        settings["risk"] = risk

    with c2:
        settings["rl"]["purge_embargo_bars"] = st.number_input(
            "Purge embargo (bars)",
            min_value=0,
            max_value=30,
            value=int(settings["rl"]["purge_embargo_bars"]),
            step=1,
        )
        wf = settings.setdefault("walk_forward", {})
        wf["n_splits"] = st.number_input(
            "Walk-forward splits",
            min_value=1,
            max_value=20,
            value=int(wf.get("n_splits", 5)),
            step=1,
        )
        wf["min_train_size"] = st.number_input(
            "Min train size (bars)",
            min_value=30,
            max_value=2000,
            value=int(wf.get("min_train_size", 252)),
            step=1,
        )
        wf["test_size"] = st.number_input(
            "Test fold size (bars)",
            min_value=10,
            max_value=500,
            value=int(wf.get("test_size", 63)),
            step=1,
        )
        settings["walk_forward"] = wf


def tab_execution(settings: dict) -> None:
    st.subheader("Execution")
    c1, c2 = st.columns(2)

    with c1:
        settings["execution"]["order_type"] = st.selectbox(
            "Order type",
            options=["limit"],
            index=0,
            help="Market orders are prohibited in this system.",
        )
        settings["execution"]["latency_bars"] = st.number_input(
            "Order latency (bars)",
            min_value=0,
            max_value=10,
            value=int(settings["execution"]["latency_bars"]),
            step=1,
        )

    with c2:
        settings["execution"]["spread_bps"] = st.number_input(
            "Synthetic spread (bps)",
            min_value=0.0,
            max_value=50.0,
            value=float(settings["execution"].get("spread_bps", 5.0)),
            step=0.5,
            help="Used when bid/ask are not present in bar data.",
        )
        st.checkbox(
            "Allow market orders",
            value=False,
            disabled=True,
            help="Hardcoded off — passive limit orders only.",
        )
        settings["execution"]["allow_market_orders"] = False


def tab_walk_forward_preview(settings: dict) -> None:
    st.subheader("Walk-forward fold preview")
    wf = settings.get("walk_forward", {})
    n_steps = st.number_input(
        "Synthetic timeline length (bars)",
        min_value=100,
        max_value=5000,
        value=int(wf.get("smoke_n_steps", 500)),
        step=50,
    )
    settings.setdefault("walk_forward", {})["smoke_n_steps"] = int(n_steps)

    splitter = PurgedWalkForwardSplitter(
        n_splits=int(wf.get("n_splits", 5)),
        embargo=int(settings["rl"]["purge_embargo_bars"]),
        min_train_size=int(wf.get("min_train_size", 252)),
        test_size=int(wf.get("test_size", 63)),
    )

    rows = []
    try:
        for fold in splitter.split(int(n_steps)):
            rows.append(
                {
                    "fold": fold.fold_id,
                    "train_bars": len(fold.train_indices),
                    "test_bars": len(fold.test_indices),
                    "test_start": int(fold.test_indices.min()),
                    "test_end": int(fold.test_indices.max()),
                    "purge_start": fold.purge_start,
                    "purge_end": fold.purge_end,
                }
            )
    except ValueError as exc:
        st.error(str(exc))
        return

    if not rows:
        st.info("No folds for the current parameters.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.bar_chart(df.set_index("fold")[["train_bars", "test_bars"]])


def tab_backtest(settings: dict) -> None:
    st.subheader("Portfolio backtest")
    bt = settings.setdefault(
        "backtest",
        {"initial_cash": 100_000.0, "rebalance_threshold": 0.05, "report_dir": "backtesting/reports"},
    )
    use_ppo = st.checkbox("Use PPO router (if trained)", value=True)
    run_exec = st.checkbox("Event-driven SPY limit-order simulation", value=False)
    bt["rebalance_threshold"] = st.slider(
        "Rebalance threshold",
        0.01,
        0.25,
        float(bt.get("rebalance_threshold", 0.05)),
        0.01,
    )
    settings["backtest"] = bt

    if st.button("Run backtest", type="primary", key="run_backtest"):
        from backtesting.run_backtest import run_portfolio_backtest

        save_settings(settings)
        with st.spinner("Running bar-by-bar portfolio backtest…"):
            try:
                result = run_portfolio_backtest(
                    settings,
                    use_ppo=use_ppo,
                    run_execution=run_exec,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        m = result.portfolio.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Information Ratio", f"{m['information_ratio']:.3f}")
        c2.metric("Total return", f"{m['total_return']:.2%}")
        c3.metric("Max drawdown", f"{m['max_drawdown']:.2%}")
        c4.metric("Avg turnover", f"{m['avg_turnover']:.3f}")

        st.line_chart(
            pd.DataFrame(
                {
                    "portfolio": result.portfolio.equity,
                    "spy": (1 + result.portfolio.spy_returns).cumprod(),
                }
            )
        )
        st.dataframe(result.portfolio.weights.tail(15), use_container_width=True)
        if result.execution_equity is not None:
            st.subheader("SPY execution equity")
            st.line_chart(result.execution_equity["equity"])


def tab_phase3_ops(settings: dict) -> None:
    st.subheader("Phase 3 — walk-forward & paper loop")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Walk-forward OOS report**")
        use_ppo_wf = st.checkbox("Use PPO (if trained)", value=False, key="wf_use_ppo")
        if st.button("Run walk-forward report", type="primary", key="run_wf_report"):
            from pipeline.walk_forward_report import run_walk_forward_report

            save_settings(settings)
            with st.spinner("Evaluating OOS folds…"):
                try:
                    result = run_walk_forward_report(
                        settings,
                        use_ppo=use_ppo_wf,
                        export_csv=True,
                    )
                except Exception as exc:
                    st.error(str(exc))
                    return
            agg = result.aggregate
            st.success(f"Report: `{result.report_path.name}`")
            st.metric("Mean IR", f"{agg['mean_information_ratio']:.3f}")
            st.metric("Mean return", f"{agg['mean_total_return']:.2%}")
            st.metric("Worst max DD", f"{agg['worst_max_drawdown']:.2%}")

    with c2:
        st.markdown("**Paper loop (dry run)**")
        use_ppo_pl = st.checkbox("Use PPO (if trained)", value=False, key="pl_use_ppo")
        if st.button("Run paper loop", key="run_paper_loop"):
            from pipeline.paper_trading_loop import run_paper_loop

            save_settings(settings)
            with st.spinner("Running offline paper loop…"):
                try:
                    result = run_paper_loop(settings, use_ppo=use_ppo_pl, dry_run=True)
                except Exception as exc:
                    st.error(str(exc))
                    return
            st.success(f"Equity {result.equity:,.0f} · report `{result.report_path.name}`")
            st.json(result.weights)


def tab_diagnostics(settings: dict) -> None:
    st.subheader("Diagnostics")
    diag = settings.setdefault("diagnostics", {})
    c1, c2 = st.columns(2)

    with c1:
        diag["audit_n_bars"] = st.number_input(
            "Audit sample bars",
            min_value=30,
            max_value=1000,
            value=int(diag.get("audit_n_bars", 120)),
            step=10,
        )
    with c2:
        diag["audit_seed"] = st.number_input(
            "Audit RNG seed",
            min_value=0,
            max_value=9999,
            value=int(diag.get("audit_seed", 42)),
            step=1,
        )
    settings["diagnostics"] = diag

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Look-ahead bias audit**")
        if st.button("Run audit", key="run_audit", use_container_width=True):
            from backtesting.event_driven_backtester import run_lookahead_bias_audit

            with st.spinner("Running audit..."):
                result = run_lookahead_bias_audit(
                    n_bars=int(diag["audit_n_bars"]),
                    seed=int(diag["audit_seed"]),
                )
            if result.passed:
                st.success("PASSED — no look-ahead violations detected.")
            else:
                st.error("FAILED")
                for msg in result.violations:
                    st.write(f"- {msg}")

    with col2:
        st.markdown("**Walk-forward smoke test**")
        if st.button("Run smoke test", key="run_smoke", use_container_width=True):
            wf = settings.get("walk_forward", {})
            with st.spinner("Running smoke test..."):
                summary = run_walk_forward_smoke_test(
                    n_steps=int(wf.get("smoke_n_steps", 500)),
                    n_splits=int(wf.get("n_splits", 3)),
                    embargo=int(settings["rl"]["purge_embargo_bars"]),
                )
            st.success(f"Completed {len(summary['folds'])} folds")
            st.dataframe(
                pd.DataFrame(summary["folds"]),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    init_state()
    render_sidebar()
    settings = st.session_state.settings

    st.title("Multi-Strategy RL Trading Bot")
    st.caption("Tune config, preview walk-forward splits, and run foundation diagnostics.")

    tabs = st.tabs(
        [
            "Universe & data",
            "Strategies",
            "Features",
            "xLSTM",
            "PPO",
            "RL & walk-forward",
            "Execution",
            "Fold preview",
            "Backtest",
            "Phase 3",
            "Diagnostics",
        ]
    )

    with tabs[0]:
        tab_universe_data(settings)
    with tabs[1]:
        tab_strategies(settings)
    with tabs[2]:
        tab_features(settings)
    with tabs[3]:
        tab_xlstm(settings)
    with tabs[4]:
        tab_ppo(settings)
    with tabs[5]:
        tab_rl(settings)
    with tabs[6]:
        tab_execution(settings)
    with tabs[7]:
        tab_walk_forward_preview(settings)
    with tabs[8]:
        tab_backtest(settings)
    with tabs[9]:
        tab_phase3_ops(settings)
    with tabs[10]:
        tab_diagnostics(settings)

    with st.expander("Raw YAML preview"):
        import yaml

        st.code(yaml.safe_dump(settings, sort_keys=False, default_flow_style=False), language="yaml")


if __name__ == "__main__":
    main()
