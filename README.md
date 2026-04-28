# 🛰️ VEGAPUNK — Quantitative Trading System

A modular, satellite-architecture algorithmic trading system for Indian equities (NSE).

## Architecture

| Satellite | Role | Status |
|-----------|------|--------|
| S1 — Market Discovery | Universe filtering | 🔲 Stub |
| S2 — NLP Sentiment | FinBERT sentiment scoring | 🔲 Stub |
| S3 — Feature Engineering | Technical + sentiment feature matrix | 🔲 Stub |
| S7 — Regime Detection | HMM market regime classification | 🔲 Stub |
| S4 — Overnight Forecaster | LSTM + XGBoost + Meta-learner ensemble | 🔲 Stub |
| S6 — Executor | Paper trading engine + risk management | 🔲 Stub |
| S5 — Adaptive Retraining | Drift detection + model governance | 🔲 Stub |

## Running the Pipeline

```bash
# Full pipeline for today
python run_pipeline.py

# Full pipeline for a specific date
python run_pipeline.py 2026-04-28
```

## Stack

- Python 3.11, PyTorch, XGBoost, hmmlearn, FinBERT
- DuckDB + Parquet for data storage
- NSE Bhavcopy + MoneyControl + ET Markets for data feeds

## Status: Phase 0 — Infrastructure