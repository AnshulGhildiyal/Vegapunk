import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S7_CFG   = CONFIG["s7"]
MODEL_DIR = Path(CONFIG["paths"]["models_incumbent"])
RAW_DIR   = Path(CONFIG["paths"]["data_raw"])


# Data Fetchers

def fetch_nifty_data(start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch Nifty 50 daily OHLCV from yfinance."""
    logger.info("[S7] Fetching Nifty 50 data from yfinance")
    df = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Close"]].rename(columns={"Close": "nifty_close"})
    df = df.dropna()
    logger.info(f"[S7] Nifty data: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_vix_data(start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch India VIX from yfinance."""
    logger.info("[S7] Fetching India VIX data")
    df = yf.download("^INDIAVIX", start=start, auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Close"]].rename(columns={"Close": "india_vix"})
    df = df.dropna()
    logger.info(f"[S7] VIX data: {len(df)} days")
    return df


def build_observation_matrix(end_date: date = None) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today()

    start = "2019-06-01"  # Extra buffer for rolling calculations

    nifty = fetch_nifty_data(start)
    vix   = fetch_vix_data(start)

    # Align on common dates
    df = nifty.join(vix, how="inner")
    df = df.sort_index()

    # Compute features
    closes = df["nifty_close"]

    # 1. 20-day return
    df["nifty_ret_20d"] = closes.pct_change(20)

    # 2. VIX already present as india_vix

    # 3. VIX 5-day change
    df["vix_change_5d"] = df["india_vix"].pct_change(5)

    # 4. Return autocorrelation (10-day rolling)
    daily_ret = closes.pct_change()
    df["ret_autocorr_10d"] = daily_ret.rolling(10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 4 else np.nan,
        raw=False
    )

    # 5. Distance from 200-day MA
    ma200 = closes.rolling(200).mean()
    df["dist_from_200dma"] = (closes - ma200) / ma200

    # Keep only rows from 2020 onward (after burn-in period)
    df = df[df.index >= "2020-01-01"]

    # Drop NaN rows
    feature_cols = [
        "nifty_ret_20d", "india_vix",
        "vix_change_5d", "ret_autocorr_10d", "dist_from_200dma"
    ]
    df = df[feature_cols].dropna()

    # Filter to end_date
    df = df[df.index.date <= end_date]

    logger.info(f"[S7] Observation matrix: {len(df)} days × {len(feature_cols)} features")
    logger.info(f"[S7] Date range: {df.index[0].date()} → {df.index[-1].date()}")

    return df


# HMM Training

def train_hmm(obs_matrix: pd.DataFrame, n_regimes: int = 3) -> tuple:
    logger.info(f"[S7] Training HMM: {n_regimes} regimes, {len(obs_matrix)} observations")

    X = obs_matrix.values.astype(float)

    # Scale features (HMM is sensitive to feature scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
        verbose=False,
    )
    model.fit(X_scaled)

    # Predict regime sequence
    regimes = model.predict(X_scaled)
    regime_series = pd.Series(regimes, index=obs_matrix.index, name="raw_regime")

    logger.success(f"[S7] HMM trained. Log-likelihood: {model.score(X_scaled):.2f}")
    logger.info(f"[S7] Regime distribution: {pd.Series(regimes).value_counts().to_dict()}")

    return model, scaler, regime_series


def label_regimes(model, scaler, obs_matrix: pd.DataFrame) -> dict:
    feature_cols = obs_matrix.columns.tolist()
    vix_idx    = feature_cols.index("india_vix")
    ret_idx    = feature_cols.index("nifty_ret_20d")

    # Get mean emissions in original scale
    means_scaled = model.means_
    means_original = scaler.inverse_transform(means_scaled)

    regime_profiles = {}
    for i in range(model.n_components):
        regime_profiles[i] = {
            "mean_ret_20d": means_original[i][ret_idx],
            "mean_vix":     means_original[i][vix_idx],
        }

    # Sort by VIX to assign labels
    sorted_by_vix = sorted(regime_profiles.items(), key=lambda x: x[1]["mean_vix"])

    label_map = {}
    label_map[sorted_by_vix[0][0]] = "TRENDING"  # Lowest VIX
    label_map[sorted_by_vix[1][0]] = "RANGING"   # Mid VIX
    label_map[sorted_by_vix[2][0]] = "CRISIS"    # Highest VIX

    logger.info("[S7] Regime labels assigned:")
    for regime_id, profile in regime_profiles.items():
        label = label_map[regime_id]
        logger.info(
            f"  Regime {regime_id} → {label}: "
            f"ret={profile['mean_ret_20d']:.3f}, vix={profile['mean_vix']:.1f}"
        )

    return label_map


def apply_stability_filter(
    regime_series: pd.Series, min_days: int = 3
) -> pd.Series:
    confirmed = regime_series.copy()
    for i in range(min_days, len(regime_series)):
        window = regime_series.iloc[i - min_days:i]
        if window.nunique() == 1:
            confirmed.iloc[i] = window.iloc[0]
        else:
            confirmed.iloc[i] = confirmed.iloc[i - 1]
    return confirmed


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model, scaler, label_map: dict):
    """Saves HMM model, scaler, and label map to models/incumbent/"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODEL_DIR / "hmm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODEL_DIR / "hmm_label_map.json", "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    logger.success(f"[S7] Model saved to {MODEL_DIR}")


def load_model():
    """Loads trained HMM from models/incumbent/"""
    with open(MODEL_DIR / "hmm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "hmm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "hmm_label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return model, scaler, label_map


# Daily Inference

def detect_regime(run_date: date) -> dict:
    """
    Loads trained HMM and returns today's regime.
    This is what the pipeline calls daily.
    """
    try:
        model, scaler, label_map = load_model()
    except FileNotFoundError:
        logger.error("[S7] No trained HMM found. Run train_and_save() first.")
        return {
            "date": run_date.isoformat(),
            "regime_label": "UNKNOWN",
            "confidence": 0.0,
            "days_in_regime": 0,
        }

    # Build recent observation matrix (last 60 days for context)
    obs = build_observation_matrix(end_date=run_date)
    if obs.empty or len(obs) < 10:
        logger.error("[S7] Insufficient data for regime detection")
        return {"date": run_date.isoformat(), "regime_label": "UNKNOWN", "confidence": 0.0}

    X_scaled = scaler.transform(obs.values.astype(float))
    raw_regimes = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    # Apply stability filter
    regime_series = pd.Series(raw_regimes, index=obs.index)
    confirmed = apply_stability_filter(regime_series)

    today_regime = int(confirmed.iloc[-1])
    today_probs  = probs[-1]
    confidence   = float(today_probs.max())
    label        = label_map.get(today_regime, "UNKNOWN")

    # Days in current regime
    days_in_regime = 1
    for i in range(len(confirmed) - 2, -1, -1):
        if confirmed.iloc[i] == today_regime:
            days_in_regime += 1
        else:
            break

    result = {
        "date":            run_date.isoformat(),
        "raw_regime":      today_regime,
        "regime_label":    label,
        "confidence":      round(confidence, 4),
        "days_in_regime":  days_in_regime,
        "transition_probs": today_probs.tolist(),
    }

    # Save to file
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"regime_{run_date.isoformat()}.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.success(
        f"[S7] Regime: {label} "
        f"(conf: {confidence:.2f}, days: {days_in_regime})"
    )
    return result


# Training Entry Point
def train_and_save():
    logger.info("[S7] Starting HMM training pipeline")

    obs = build_observation_matrix()
    model, scaler, regime_series = train_hmm(obs, n_regimes=3)
    label_map = label_regimes(model, scaler, obs)
    confirmed = apply_stability_filter(regime_series)

    save_model(model, scaler, label_map)

    # Print historical regime summary
    confirmed_labels = confirmed.map(label_map)
    print(f"\n{'='*55}")
    print("REGIME DETECTION — HISTORICAL SUMMARY")
    print(f"{'='*55}")
    print(f"Date range : {obs.index[0].date()} → {obs.index[-1].date()}")
    print(f"Total days : {len(obs)}")
    print(f"\nRegime distribution:")
    print(confirmed_labels.value_counts().to_string())
    print(f"\nRecent 10 days:")
    recent = pd.DataFrame({
        "regime": confirmed_labels.tail(10),
    })
    print(recent.to_string())
    print(f"{'='*55}\n")

    return model, scaler, label_map, confirmed


if __name__ == "__main__":
    train_and_save()