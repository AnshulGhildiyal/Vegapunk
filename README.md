# 🛰️ VEGAPUNK

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regime--Conditional-orange)](https://xgboost.readthedocs.io)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-HMM%20Regime-green)](https://hmmlearn.readthedocs.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-Parquet%20Cache-yellow)](https://duckdb.org)
[![NSE Equities](https://img.shields.io/badge/Market-NSE%20Equities-blue)](https://nseindia.com)
[![Paper Trading Live](https://img.shields.io/badge/Paper%20Trading-LIVE-brightgreen)]()

**Algorithmic swing trading system for NSE equities.**  
7 modular satellites handle the full pipeline: universe filtering → sentiment scoring → feature engineering → regime detection → forecasting → execution → monitoring. Running live paper trading since May 2026.

---

## Live Performance

| Metric | Value |
|--------|-------|
| Portfolio | ₹5,03,041 |
| Return | +0.61% (started May 4, 2026) |
| Open Positions | 22 |
| Regime | TRENDING (conf: 1.00) |
| Directional Accuracy (10d) | 59.03% |
| Max-Hold Win Rate | 58% |
| Last Updated | 2026-05-29 |

> **Context:** +0.61% portfolio return — first time in positive territory since launch. Recovered from a May 12 black swan event (India-Pakistan military tensions + FII outflows) where Nifty fell 8%. Max-hold exit win rate stable at 58%. Directional accuracy recovering to 54.35%, above the 52% threshold.

---

## Architecture

The system is built as 7 independent satellites that each own one responsibility. Data flows in one direction — no circular dependencies, no shared state except the parquet cache.

```
S1 Universe                S2 Sentiment
┌─────────────┐            ┌──────────────────┐
│ 5-gate      │            │ NSE Announcements │
│ liquidity   │            │ + Google News RSS │
│ filter      │            │ → FinBERT (local) │
│ 126 stocks  │            │ 5 features        │
└──────┬──────┘            └────────┬─────────┘
       │                            │
       └──────────┬─────────────────┘
                  ▼
          S3 Feature Engineering
          ┌─────────────────────┐
          │ 27 features across  │
          │ 4 families:         │
          │ price / volume /    │
          │ technical / cross-  │
          │ sectional + sentiment│
          └──────────┬──────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  S7 Regime Detection     S4 Forecaster
  ┌──────────────────┐    ┌───────────────────┐
  │ 4-state Gaussian │    │ Regime-conditional │
  │ HMM on Nifty +   │───▶│ XGBoost           │
  │ India VIX        │    │ 18mo walk-forward  │
  │ TRENDING /       │    │ OOF acc: 0.5539   │
  │ RANGING / CRISIS │    └────────┬──────────┘
  └──────────────────┘             │
                                   ▼
                            S6 Executor
                            ┌──────────────────┐
                            │ Half-Kelly sizing │
                            │ Indian cost model │
                            │ Circuit breakers  │
                            │ L1/L2/L3 DD gates │
                            └────────┬─────────┘
                                     │
                                     ▼
                              S5 Monitor
                              ┌──────────────────┐
                              │ PSI drift detect  │
                              │ Directional acc   │
                              │ Auto-retrain      │
                              └──────────────────┘
```

---

## Why Satellite Architecture?

A monolithic trading system fails silently — one bad data fetch corrupts everything downstream. The satellite pattern isolates failures: if S2 sentiment scoring fails, S3 fills zeros and the pipeline continues. Each satellite can be tested, replaced, or upgraded independently. This matters in production where FinBERT (S2) will eventually be replaced by an MLX on-device model without touching S4, S6, or S7.

---

## Satellites

### S1 — Universe Filter
Filters 148 NSE symbols down to ~126 tradeable stocks daily using 5 gates: liquidity (ADTV ≥ ₹5Cr), price band (₹50–₹5000), listing age, ASM/GSM surveillance exclusion, and sector concentration cap. **Engineering decision:** 90-day rolling ADTV cache in Parquet avoids repeat downloads and handles NSE's inconsistent symbol naming via a 21-symbol blocklist.

### S2 — Sentiment Engine
Fetches NSE corporate announcements (253–886 per day) and Google News RSS headlines, then scores them using ProsusAI/FinBERT running locally on CPU. Produces 5 features: raw sentiment, momentum, volume, volatility, and price-sentiment divergence. **Engineering decision:** FinBERT runs locally (no API key, no rate limits, no cost) and processes 126 stocks in ~30 seconds by only calling Google News for stocks with active NSE announcements that day.

### S3 — Feature Engineering
Builds a 27-feature matrix across 4 families: price returns (ret_1d/5d/10d/20d, gap_open, dist_52w_high), volume (OBV slope, volume ratio, price-volume correlation), technical (RSI-14, MACD histogram, Bollinger position, ATR-14, MA cross), and cross-sectional (universe momentum rank, excess returns, volatility rank). **Engineering decision:** Leakage audit ensured no forward-looking features — all features computed using only data available at close on the signal date.

### S7 — Regime Detection
4-state Gaussian HMM trained on 5 Nifty/VIX-derived features: 20-day return, India VIX level, VIX 5-day change, return autocorrelation, and distance from 200 DMA. Correctly flagged COVID March 2020 as CRISIS and the May 2026 India-Pakistan tensions as RANGING. **Engineering decision:** 4th HMM state merged into TRENDING to prevent sparse-class instability — the model now produces 3 clean regime labels with conf scores.

### S4 — Forecaster
Regime-conditional XGBoost: separate models for TRENDING and RANGING regimes, selected via walk-forward validation (18-month train / 3-month val / 5-day embargo). Training data: 2022–2026, 134,267 samples, 1,129 days. OOF accuracy 0.5539 overall, 0.5674 TRENDING. **Engineering decision:** Walk-forward with embargo prevents data leakage between train and validation that would inflate accuracy metrics on time-series data.

### S6 — Executor
Full Indian cost model (brokerage ₹20, STT 0.1%, exchange fees, SEBI, GST, stamp duty, 0.07% slippage). Half-Kelly position sizing (0.5× normal, 0.25× in crisis). Circuit breakers at L1=1.5% daily drawdown, L2=3%, L3=7% weekly. **Engineering decision:** Half-Kelly rather than full Kelly — trading is not a repeated identical game, and half-Kelly significantly reduces ruin probability while preserving most of the growth rate.

### S5 — Monitor
Three monitoring loops: directional accuracy against 10-day realized outcomes, Population Stability Index drift detection across all 27 features, and portfolio health checks. Triggers automatic XGBoost retraining when 60%+ of features show PSI drift. **Engineering decision:** PSI threshold set at 0.25 per feature, breach trigger at 60% — avoids false retrain signals from normal short-term distribution shifts while catching genuine regime changes.

---

## Data Layer

```
data/
├── raw/
│   ├── bhavcopy/          # 1,579 daily Parquet files (2020–2026)
│   │   └── YYYY-MM-DD.parquet
│   └── sentiment/
│       ├── nse_announcements/   # NSE corporate filings (cached)
│       └── scores/              # FinBERT score JSONs per date
├── processed/
│   ├── features_YYYY-MM-DD.parquet   # 27-feature matrices
│   └── regime_YYYY-MM-DD.json        # HMM regime labels
├── universes/
│   └── universe_YYYY-MM-DD.parquet   # Daily approved stock lists
└── trades/
    └── portfolio_state.json           # Live paper trading state
```

**1,579 trading days cached (2020–2026).** All intermediate data is Parquet for columnar compression and fast symbol-level filtering. No database required — DuckDB-style queries run directly on the Parquet files via pandas.

---

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11 | Type hints, walrus operator, speed |
| ML | XGBoost + hmmlearn | Gradient boosting for tabular, HMM for regime |
| Sentiment | ProsusAI/FinBERT | Finance-specific BERT, runs on CPU |
| Data | yfinance + NSE API | Free, reliable NSE data sources |
| Storage | Parquet + JSON | Columnar, compressed, zero-config |
| Logging | loguru | Structured logs with levels |
| Config | YAML | Single source of truth for all params |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 — Infrastructure | ✅ Complete | 7-satellite architecture, Parquet cache, walk-forward |
| Phase 1 — Backtesting | ✅ Complete | 2022–2026 backtest, OOS validation, regime-conditional XGB |
| Phase 2 — FinBERT Sentiment | ✅ Complete | Local FinBERT, NSE announcements, Google News |
| Phase 3 — Live Execution | ✅ Running | Paper trading since May 4, 2026 |
| Phase 4 — On-Device MLX | 🔲 Sep 2026 | Apple Silicon MLX models, full sentiment backfill |
| Phase 5 — Live Execution | 🔲 Oct 2026 | Zerodha Kite API, intraday strategies |
| Phase 6 — LSTM Ensemble | 🔲 2027 | LSTM + XGBoost meta-learner stacking |

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Daily pipeline (run after 3:30 PM IST)
python run_pipeline.py

# Dashboard
python dashboard/daily_report.py

# Retrain models
python satellites/s4_forecaster/xgb_model.py

# Backtest
python scripts/backtest.py
```

---

## Key Config Parameters

```yaml
s6:
  stop_loss_pct: 0.04          # 4% stop-loss
  max_hold_days: 10            # 10-day maximum hold
  kelly_fraction: 0.5          # Half-Kelly sizing
  max_single_position_pct: 0.05 # 5% portfolio cap per position

s4:
  prediction_horizon_days: 5   # 5-day forward return target
  train_months: 18             # Walk-forward training window
  val_months: 3                # Walk-forward validation window
  embargo_days: 5              # Gap between train and val

s7:
  n_regimes: 4                 # HMM states (merged to 3 labels)
```

---

*Built by Anshul Ghildiyal — MCA student, UPES Dehradun (2026–2028)*
