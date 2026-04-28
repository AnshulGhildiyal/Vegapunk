
import yaml
from datetime import date
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

with open("config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


def run_s1(run_date: str) -> dict:
    """S1 — Universe Filter"""
    logger.info(f"[S1] Building universe for {run_date}")
    return {"status": "STUB", "n_stocks": 0, "universe_path": None}

def run_s2(run_date: str, universe: dict) -> dict:
    """S2 — NLP Sentiment Engine"""
    logger.info(f"[S2] Scoring sentiment for {run_date}")
    return {"status": "STUB", "n_articles": 0, "sentiment_path": None}

def run_s3(run_date: str, universe: dict, sentiment: dict) -> dict:
    """S3 — Feature Engineering"""
    logger.info(f"[S3] Building feature matrix for {run_date}")
    return {"status": "STUB", "lstm_shape": None, "xgb_shape": None}

def run_s7(run_date: str, features: dict) -> dict:
    """S7 — Regime Detection"""
    logger.info(f"[S7] Detecting market regime for {run_date}")
    return {"status": "STUB", "regime_label": "UNKNOWN", "confidence": 0.0}

def run_s4(run_date: str, features: dict, regime: dict) -> dict:
    """S4 — Overnight Forecaster"""
    logger.info(f"[S4] Generating signals for {run_date}")
    return {"status": "STUB", "n_signals": 0, "signals_path": None}

def run_s6(run_date: str, signals: dict, regime: dict) -> dict:
    """S6 — Executor (Paper Trading)"""
    logger.info(f"[S6] Executing trades for {run_date}")
    return {"status": "STUB", "trades_entered": 0, "trades_exited": 0}

def run_s5(run_date: str) -> dict:
    """S5 — Adaptive Retraining Monitor"""
    logger.info(f"[S5] Running monitoring checks for {run_date}")
    return {"status": "STUB", "retrain_triggered": False}


def run_pipeline(run_date: str = None):
    if run_date is None:
        run_date = date.today().isoformat()

    logger.info("=" * 60)
    logger.info(f"VEGAPUNK Pipeline — {run_date}")
    logger.info("=" * 60)

    # Data flow: S1 → S2 → S3 → S7 → S4 → S6 → S5
    universe  = run_s1(run_date)
    sentiment = run_s2(run_date, universe)
    features  = run_s3(run_date, universe, sentiment)
    regime    = run_s7(run_date, features)
    signals   = run_s4(run_date, features, regime)
    execution = run_s6(run_date, signals, regime)
    monitoring = run_s5(run_date)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete for {run_date}")
    logger.info(f"  Regime   : {regime['regime_label']} (conf: {regime['confidence']:.2f})")
    logger.info(f"  Signals  : {signals['n_signals']}")
    logger.info(f"  Trades   : {execution['trades_entered']} entered, {execution['trades_exited']} exited")
    logger.info(f"  Retrain  : {'TRIGGERED' if monitoring['retrain_triggered'] else 'NOT TRIGGERED'}")
    logger.info("=" * 60)

    return {
        "date": run_date,
        "universe": universe,
        "regime": regime,
        "signals": signals,
        "execution": execution,
    }

if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(date_arg)