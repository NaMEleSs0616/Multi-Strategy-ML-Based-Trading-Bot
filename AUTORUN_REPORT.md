# AUTORUN_REPORT.md

**Generated:** 2026-05-27 (automated AFK run)  
**Working directory:** `/Users/deborshi/Multi-Strategy ML based trading bot`  
**Python environment:** `.venv`  
**PYTHONPATH:** `.`  
**SKIP_EDGAR_SENTIMENT:** `1`

---

## Summary Table

| # | Step | Status | Duration (s) | Notes |
|---|------|--------|-------------|-------|
| 1 | `python scripts/sync_data.py` | ✅ PASS | ~19 | 24 series synced (8 symbols × 3 intervals: 1d, 5m, 15m); 500 rows per daily, 4680 per 5m, 1560 per 15m |
| 2 | `python scripts/train_xlstm.py` | ✅ PASS | ~24 | 20 epochs; best_val_mse=0.067686; checkpoint frozen |
| 3 | `python scripts/train_ppo.py` | ✅ PASS | ~33 | 3 walk-forward folds; checkpoint saved |
| 4 | `python scripts/run_backtest.py` | ✅ PASS | ~18 | Total return 9.69%; IR -0.849; max DD -4.05% |
| 5 | `python scripts/run_backtest.py --equal-weight` | ✅ PASS | ~5 | Same output as step 4 (equal-weight uses same path currently) |
| 6 | `python scripts/run_walk_forward_report.py` | ✅ PASS | ~18 | 3 folds OOS; mean IR -1.18; mean return 2.70%; worst DD -7.16% |
| 7 | `python scripts/run_paper_loop.py` (dry run, killed after 20s) | ✅ PASS | ~20 | 1 order/fill; equity 100,025; dry-run mode confirmed |
| 8 | `python -m backtesting.event_driven_backtester` | ✅ PASS | ~3 | **Look-ahead bias audit: PASSED** |
| 9 | `pytest tests/ -q -m "not slow"` | ⚠️ PARTIAL | ~5 | 41 passed, **2 failed**, 1 deselected |
| 10 | `pytest tests/ -q -m "slow"` | ✅ PASS | ~4 | 1 passed, 43 deselected |

**Overall: 8/10 steps fully passed; 1 partial (2 test failures with known root cause).**

---

## Artifacts Created / Updated

| Path | Size | Description |
|------|------|-------------|
| `data/storage/market_data.db` | 6.9 MB | SQLite market data (8 symbols × 3 intervals) |
| `models/xlstm/checkpoints/xlstm_frozen.pt` | 1.0 MB | Frozen xLSTM model weights |
| `models/xlstm/checkpoints/xlstm_best.pt` | 46 KB | Best validation checkpoint |
| `models/xlstm/checkpoints/embeddings.npy` | 223 KB | Pre-computed xLSTM embeddings |
| `models/xlstm/checkpoints/train_history.json` | 730 B | Training loss history |
| `models/ppo/checkpoints/ppo_router.zip` | 229 KB | Stable-Baselines3 PPO strategy router |
| `models/ppo/checkpoints/train_summary.json` | 215 B | PPO training summary |
| `backtesting/reports/latest_backtest.json` | 324 B | Full backtest metrics |
| `backtesting/reports/walk_forward_report.json` | 1.7 KB | Walk-forward OOS report (3 folds) |
| `backtesting/reports/walk_forward_folds/` | dir | Per-fold CSV files |
| `backtesting/reports/paper_loop_latest.json` | 313 B | Paper loop dry-run snapshot |

---

## Key Backtest Metrics

### Full-sample backtest (`run_backtest.py`)
| Metric | Value |
|--------|-------|
| Total return | **9.69%** |
| SPY return (benchmark) | 40.94% |
| Information Ratio | -0.849 |
| Max Drawdown | -4.05% |
| Annualized Volatility | 6.45% |
| Avg Daily Turnover | 2.10% |
| N bars | 445 |
| Policy | equal_weight |

### Walk-forward OOS (3 folds, PPO policy)
| Fold | Test Period | Total Return | IR | Max DD |
|------|------------|-------------|-----|--------|
| 0 | 2025-08-19 → 2025-11-14 | +2.20% | -0.861 | -0.74% |
| 1 | 2025-11-24 → 2026-02-25 | +0.37% | -1.834 | -0.37% |
| 2 | 2026-03-05 → 2026-05-26 | +5.52% | -0.849 | -7.16% |
| **Mean** | | **+2.70%** | **-1.181** | **-7.16% (worst)** |

> **Interpretation:** The bot is profitable in absolute terms but underperforms SPY significantly. The negative IR across all folds indicates the Sharpe-weighted excess return is consistently below the risk-free rate. Fold 2 shows improving momentum (+5.52%) but also the highest drawdown. The strategy needs either a stronger alpha signal or macro regime filtering (see Next Steps).

---

## Test Results

**Fast suite (`not slow`):** 41 passed, 2 failed  
**Slow suite:** 1 passed

### Failures (2) — Root cause: test DB date mismatch

Both failures are in the same underlying function `align_to_feature_index` in `pipeline/training_data.py`.

**Tests affected:**
- `tests/test_paper_loop.py::test_paper_loop_dry_run`
- `tests/test_walk_forward_report.py::test_walk_forward_report_equal_weight`

**Root cause:** The test fixture `seed_market_db` seeds synthetic price data with dates in **2023-01** through **2023-10**, but the xLSTM `embeddings.npy` checkpoint on disk was trained on **real market data (2024-08 → 2026-05)**. When the test loads the live checkpoint into a test context with synthetic DB dates, the date alignment finds **0 overlapping bars** and raises:

```
ValueError: Insufficient aligned bars (0) for training
```

**Last 20 lines of error (test_paper_loop):**
```
pipeline/training_data.py:50: ValueError
ValueError: Insufficient aligned bars (0) for training

  feat_index = DatetimeIndex(['2024-08-15', ..., '2026-05-26'], length=445)
  bundle.master_index = DatetimeIndex(['2023-01-01', ..., '2023-10-27'], length=300)
  -> 0 overlapping dates
```

**Fix required:** The tests should either (a) mock the checkpoint loader to return a dummy embedding whose index matches the seeded DB dates, or (b) add a `checkpoint_dir=None` / `use_real_checkpoints=False` flag to the pipeline functions to generate random embeddings for tests. This is a **test isolation issue, not a production bug** — the pipeline itself runs correctly end-to-end (steps 4–8 all pass).

---

## Next Steps (when you add API keys)

### 1. Add FRED_API_KEY — Macro features
```bash
export FRED_API_KEY="your_fred_key"
# Unset the skip flag to enable EDGAR/FRED enrichment
unset SKIP_EDGAR_SENTIMENT
python scripts/sync_data.py          # now fetches macro series (CPI, FFR, yield curve)
python scripts/train_xlstm.py        # retrain with macro features in input
python scripts/train_ppo.py          # retrain PPO router with macro-enriched embeddings
python scripts/run_backtest.py       # expect IR improvement with regime awareness
```

### 2. Add Alpaca paper trading keys — Live paper loop
```bash
export APCA_API_KEY_ID="your_alpaca_key_id"
export APCA_API_SECRET_KEY="your_alpaca_secret"
export APCA_BASE_URL="https://paper-api.alpaca.markets"   # paper trading endpoint

# Run paper trading loop (live quotes, real order simulation)
python scripts/run_paper_loop.py

# Schedule as a daily cron (after market open) e.g.:
# 0 10 * * 1-5 cd /path/to/bot && source .venv/bin/activate && \
#   APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python scripts/run_paper_loop.py
```

### 3. Fix the 2 test failures (date alignment)
The fastest fix is to patch `seed_market_db` to use dates that overlap with current real embeddings, or pass `require_encoder=False` and skip the checkpoint load in unit tests:
```python
# In tests/conftest.py or each test — seed dates matching real embedding range
# seed_market_db(db, start="2024-09-01", end="2026-05-01")
```

### 4. Investigate negative Information Ratio
The IR is consistently negative across all OOS folds. Consider:
- Adding momentum or trend-following signal to counteract mean-reversion bias
- Enabling FRED macro regime filter (see step 1 above)
- Tuning PPO reward function to penalize under-performance vs SPY rather than raw PnL

### 5. Rerun full pipeline with keys
```bash
cd "/Users/deborshi/Multi-Strategy ML based trading bot"
source .venv/bin/activate
export FRED_API_KEY="..."
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
export PYTHONPATH=.
# Full pipeline
python scripts/sync_data.py
python scripts/train_xlstm.py
python scripts/train_ppo.py
python scripts/run_backtest.py
python scripts/run_walk_forward_report.py
python scripts/run_paper_loop.py
pytest tests/ -q
```

---

## Miscellaneous Notes

- **Matplotlib cache warning** appears on all GPU-less runs due to `/Users/deborshi/.matplotlib` permissions. This is cosmetic and does not affect functionality. Fix with: `mkdir -p ~/matplotlib-cache && export MPLCONFIGDIR=~/matplotlib-cache`
- The `--equal-weight` flag on `run_backtest.py` currently produces identical results to the default run (same JSON report path). The flag may not yet switch to a separate weighting path — worth investigating in `scripts/run_backtest.py`.
- `run_paper_loop.py` does not accept `--dry-run`; it runs in dry-run mode automatically (no real Alpaca keys present). The script is safe to run without credentials.

---

*Report generated by automated AFK pipeline runner. All steps completed without manual intervention.*
