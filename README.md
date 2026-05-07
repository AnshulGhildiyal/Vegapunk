# 🛰️ VEGAPUNK — Quantitative Trading System

A modular, satellite-architecture algorithmic trading system for Indian equities (NSE).

## Live Status (Day 10)

| Satellite | Role | Status | Notes |
|-----------|------|--------|-------|
| S1 — Market Discovery | Universe filtering | 🟢 LIVE | 126 stocks, 2 gates active |
| S2 — NLP Sentiment | FinBERT sentiment | ⬜ STUB | Phase 2 |
| S3 — Feature Engineering | 27-feature matrix | 🟢 LIVE | 4 families |
| S7 — Regime Detection | HMM classification | 🟢 LIVE | RANGING (conf=1.00) |
| S4 — Overnight Forecaster | XGBoost ensemble | 🟢 LIVE | OOF acc=0.5438 |
| S6 — Executor | Paper trading | 🟢 LIVE | ₹5,09,583 (+1.92%) |
| S5 — Retraining | Drift monitoring | 🟢 LIVE | Acc=0.77, no drift |

## Paper Trading Performance
- **Started:** May 4, 2026
- **Portfolio:** ₹5,09,583 (+1.92%)
- **Open Positions:** 28
- **Regime:** RANGING

## Daily Workflow
```bash
python run_pipeline.py          # Run after market close
python dashboard/daily_report.py # View portfolio state
```

## Stack
- Python 3.11, XGBoost, hmmlearn, yfinance
- DuckDB + Parquet for data storage
- 1,570 trading days cached (2020–2026)

## Status: Phase 0 — Infrastructure