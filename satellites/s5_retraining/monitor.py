import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S5_CFG   = CONFIG["s5"]
PROC_DIR = Path(CONFIG["paths"]["data_processed"])
TRADE_DIR = Path("data/trades")


# Loop 1: Daily Error Monitor

def compute_directional_accuracy(lookback_days: int = 10) -> dict:
    today = date.today()
    results = []

    for i in range(5, lookback_days + 5):
        signal_date = today - timedelta(days=i)
        outcome_date = signal_date + timedelta(days=5)

        signal_path = PROC_DIR / f"features_{signal_date.isoformat()}.parquet"
        outcome_path = PROC_DIR / f"features_{outcome_date.isoformat()}.parquet"

        if not signal_path.exists() or not outcome_path.exists():
            continue

        signals  = pd.read_parquet(signal_path)[["symbol", "close"]].rename(
            columns={"close": "entry_close"}
        )
        outcomes = pd.read_parquet(outcome_path)[["symbol", "close"]].rename(
            columns={"close": "exit_close"}
        )

        merged = signals.merge(outcomes, on="symbol", how="inner")
        merged["actual_ret"] = np.log(
            merged["exit_close"] / merged["entry_close"]
        )
        merged["actual_direction"] = (merged["actual_ret"] > 0).astype(int)
        merged["signal_date"] = signal_date.isoformat()
        results.append(merged)

    if not results:
        return {"status": "insufficient_data", "accuracy": None}

    all_results = pd.concat(results)
    accuracy = all_results["actual_direction"].mean()

    logger.info(f"[S5] Directional accuracy ({lookback_days}d): {accuracy:.4f}")

    threshold = S5_CFG["directional_accuracy_threshold"]
    if accuracy < threshold:
        logger.warning(
            f"[S5]  Accuracy {accuracy:.4f} below threshold {threshold} "
            f"— retraining may be needed"
        )

    return {
        "status":   "ok",
        "accuracy": round(accuracy, 4),
        "n_samples": len(all_results),
        "threshold": threshold,
        "flag":      accuracy < threshold,
    }


#  Loop 2: Feature Drift (PSI)

def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    ref_counts, edges = np.histogram(reference, bins=buckets)
    cur_counts, _     = np.histogram(current, bins=edges)

    # Avoid zeros
    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_feature_drift(reference_days: int = 60, current_days: int = 10) -> dict:
    """
    Compares recent feature distributions against historical reference.
    Flags features with PSI >= 0.2.
    """
    today = date.today()

    # Load reference period
    ref_frames = []
    for i in range(current_days + 1, current_days + reference_days + 1):
        path = PROC_DIR / f"features_{(today - timedelta(days=i)).isoformat()}.parquet"
        if path.exists():
            ref_frames.append(pd.read_parquet(path))

    # Load current period
    cur_frames = []
    for i in range(1, current_days + 1):
        path = PROC_DIR / f"features_{(today - timedelta(days=i)).isoformat()}.parquet"
        if path.exists():
            cur_frames.append(pd.read_parquet(path))

    if not ref_frames or not cur_frames:
        logger.warning("[S5] Insufficient data for drift detection")
        return {"status": "insufficient_data"}

    ref_df = pd.concat(ref_frames)
    cur_df = pd.concat(cur_frames)

    feature_cols = [
        c for c in ref_df.columns
        if c not in ["symbol", "date", "close"]
        and ref_df[c].dtype in [np.float64, np.float32, np.int64]
    ]

    psi_results = {}
    drifted = []

    for col in feature_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values

        if len(ref_vals) < 20 or len(cur_vals) < 10:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        psi_results[col] = round(psi, 4)

        if psi >= S5_CFG["psi_drift_threshold"]:
            drifted.append(col)

    drift_pct = len(drifted) / len(psi_results) if psi_results else 0

    logger.info(
        f"[S5] Feature drift: {len(drifted)}/{len(psi_results)} features "
        f"above PSI threshold ({drift_pct:.1%})"
    )

    retrain_flag = drift_pct >= S5_CFG["psi_feature_breach_pct"]
    if retrain_flag:
        logger.warning(f"[S5] 🔄 Drift threshold exceeded — retraining recommended")

    return {
        "status":        "ok",
        "n_features":    len(psi_results),
        "n_drifted":     len(drifted),
        "drift_pct":     round(drift_pct, 4),
        "drifted_cols":  drifted[:5],  # Top 5 most drifted
        "retrain_flag":  retrain_flag,
    }


# Loop 3: Portfolio Health 

def check_portfolio_health() -> dict:
    """Reads portfolio state and checks for anomalies."""
    state_file = TRADE_DIR / "portfolio_state.json"

    if not state_file.exists():
        return {"status": "no_trades_yet"}

    with open(state_file) as f:
        state = json.load(f)

    trade_log = state.get("trade_log", [])
    daily_log  = state.get("daily_log", [])

    if not trade_log:
        return {"status": "no_trades_yet"}

    # Closed trades only (have exit_price)
    closed = [t for t in trade_log if "exit_price" in t]

    if closed:
        pnls      = [t["pnl"] for t in closed]
        win_rate  = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl   = np.mean(pnls)
        total_pnl = sum(pnls)
    else:
        win_rate = avg_pnl = total_pnl = 0

    latest = daily_log[-1] if daily_log else {}

    result = {
        "status":           "ok",
        "portfolio_value":  latest.get("portfolio_value", 0),
        "total_return_pct": latest.get("total_return_pct", 0),
        "open_positions":   latest.get("n_positions", 0),
        "total_trades":     len(closed),
        "win_rate":         round(win_rate, 4),
        "avg_pnl":          round(avg_pnl, 2),
        "total_pnl":        round(total_pnl, 2),
        "circuit_breaker":  state.get("circuit_breaker", "CLEAR"),
    }

    logger.info(
        f"[S5] Portfolio: ₹{result['portfolio_value']:,.0f} | "
        f"return={result['total_return_pct']:+.2f}% | "
        f"trades={result['total_trades']} | "
        f"win_rate={result['win_rate']:.1%}"
    )

    return result


# Main Monitor
def run_monitoring() -> dict:
    """Runs all monitoring loops. Called nightly by pipeline."""
    logger.info("[S5] Running monitoring checks")

    accuracy = compute_directional_accuracy(lookback_days=10)
    drift    = check_feature_drift()
    health   = check_portfolio_health()

    retrain_triggered = (
        accuracy.get("flag", False) or
        drift.get("retrain_flag", False)
    )

    if retrain_triggered:
        logger.warning("[S5] 🔄 RETRAINING TRIGGERED")
    else:
        logger.info("[S5] ✅ All checks passed — no retraining needed")

    return {
        "accuracy":          accuracy,
        "drift":             drift,
        "portfolio_health":  health,
        "retrain_triggered": retrain_triggered,
    }


if __name__ == "__main__":
    result = run_monitoring()
    print(f"\nRetrain triggered: {result['retrain_triggered']}")