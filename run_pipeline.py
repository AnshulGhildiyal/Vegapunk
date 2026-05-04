
import yaml
from datetime import date
from loguru import logger
import sys
from satellites.s1_universe.universe_builder import build_universe
from satellites.s3_features.feature_engineer import build_feature_matrix
from satellites.s7_regime.regime_detector import detect_regime, train_and_save
from satellites.s4_forecaster.xgb_model import predict_today
from satellites.s6_executor.paper_trader import PaperTrader
from pathlib import Path
from datetime import date as date_type
import pandas as pd

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

with open("config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

_paper_trader = None

def run_s1(run_date: str) -> dict:
    logger.info(f"[S1] Building universe for {run_date}")
    dt = date_type.fromisoformat(run_date)

    # During development, always resolve to last available trading day
    from satellites.s1_universe.universe_builder import build_universe, get_last_trading_day
    trading_day = get_last_trading_day(dt)
    if trading_day is None:
        return {"status": "FAILED", "n_stocks": 0, "universe": None}

    universe = build_universe(trading_day)
    if universe is None:
        return {"status": "FAILED", "n_stocks": 0, "universe": None}

    return {
        "status": "OK",
        "n_stocks": len(universe),
        "universe": universe,
        "universe_path": f"data/universes/universe_{trading_day.isoformat()}.parquet"
    }

def run_s2(run_date: str, universe: dict) -> dict:
    """S2 — NLP Sentiment Engine"""
    logger.info(f"[S2] Scoring sentiment for {run_date}")
    return {"status": "STUB", "n_articles": 0, "sentiment_path": None}

def run_s3(run_date: str, universe: dict, sentiment: dict) -> dict:
    """S3 — Feature Engineering"""
    logger.info(f"[S3] Building feature matrix for {run_date}")

    if universe.get("status") != "OK" or universe.get("universe") is None:
        logger.warning("[S3] No universe available — skipping feature engineering")
        return {"status": "SKIPPED", "features": None}

    dt = date_type.fromisoformat(run_date)
    result = build_feature_matrix(universe["universe"], dt)

    if result is None:
        return {"status": "FAILED", "features": None}

    return {
        "status":     "OK",
        "features":   result["xgb"],
        "n_stocks":   result["n_stocks"],
        "n_features": result["n_features"],
        "path":       result["path"],
    }

def run_s7(run_date: str, features: dict) -> dict:
    """S7 — Regime Detection"""
    logger.info(f"[S7] Detecting market regime for {run_date}")
    dt = date_type.fromisoformat(run_date)

    model_path = Path("models/incumbent/hmm_model.pkl")
    if not model_path.exists():
        logger.warning("[S7] No trained model found — training now")
        train_and_save()

    regime = detect_regime(dt)
    return regime


def run_s4(run_date: str, features: dict, regime: dict) -> dict:
    """S4 — Overnight Forecaster"""
    logger.info(f"[S4] Generating signals for {run_date}")

    if features.get("status") != "OK" or features.get("features") is None:
        return {"status": "SKIPPED", "n_signals": 0, "signals": None}

    regime_label = regime.get("regime_label", "RANGING")
    if regime_label == "UNKNOWN":
        regime_label = "RANGING"

    features_df = features["features"]
    signals = predict_today(features_df, regime_label)

    if signals is None or signals.empty:
        return {"status": "FAILED", "n_signals": 0, "signals": None}

    # Top signals only
    top_longs  = signals[signals["direction"] == "LONG"].head(20)
    top_shorts = signals[signals["direction"] == "SHORT"].tail(20)
    top_signals = pd.concat([top_longs, top_shorts])

    logger.info(
        f"[S4] Top signals: {len(top_longs)} LONG, "
        f"{len(top_shorts)} SHORT"
    )
    logger.info(
        f"[S4] Top 5 LONG: "
        f"{top_longs['symbol'].head(5).tolist()}"
    )

    return {
        "status":    "OK",
        "n_signals": len(top_signals),
        "signals":   top_signals,
        "regime":    regime_label,
    }

def run_s6(run_date: str, signals: dict, regime: dict) -> dict:
    """S6 — Paper Trading Executor"""
    logger.info(f"[S6] Executing trades for {run_date}")

    trader = get_trader()

    # Get current prices from universe
    prices_path = Path(f"data/universes/universe_{run_date}.parquet")

    # Try yesterday's universe if today's not available
    if not prices_path.exists():
        universe_files = sorted(Path("data/universes").glob("universe_*.parquet"))
        if universe_files:
            prices_path = universe_files[-1]

    if not prices_path.exists():
        logger.warning("[S6] No universe/price data — skipping execution")
        return {"status": "SKIPPED", "trades_entered": 0, "trades_exited": 0}

    universe_df = pd.read_parquet(prices_path)
    current_prices = dict(zip(universe_df["symbol"], universe_df["close"]))

    regime_label = regime.get("regime_label", "RANGING")
    signal_df    = signals.get("signals")

    result = trader.run_day(
        signals        = signal_df,
        current_prices = current_prices,
        regime         = regime_label,
        run_date       = run_date,
    )

    return {
        "status":        "OK",
        "trades_entered": result["entries"],
        "trades_exited":  result["exits"],
        "portfolio_value": result["portfolio_value"],
        "total_return":    result["total_return"],
        "circuit_breaker": result["circuit_breaker"],
    }

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