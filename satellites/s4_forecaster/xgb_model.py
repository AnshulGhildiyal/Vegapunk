import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss
)
import config

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S4_CFG    = CONFIG["s4"]
MODEL_DIR = Path(CONFIG["paths"]["models_incumbent"])
RAW_DIR   = Path(CONFIG["paths"]["data_raw"]) / "bhavcopy"
PROC_DIR  = Path(CONFIG["paths"]["data_processed"])


# XGBoost Config

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.02,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 30,
    "reg_lambda":       1.5,
    "reg_alpha":        0.1,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

REGIME_LABELS = {0: "TRENDING", 1: "RANGING", 2: "CRISIS"}
REGIME_IDS    = {v: k for k, v in REGIME_LABELS.items()}


# Data Loading

def load_all_features(start_date=date(2022, 1, 1), end_date=None):
    if end_date is None:
        end_date = date.today() - timedelta(days=10)
    frames = []
    current = pd.Timestamp(start_date)
    end     = pd.Timestamp(end_date)

    while current <= end:
        path = PROC_DIR / f"features_{current.strftime('%Y-%m-%d')}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames).drop_duplicates(subset=["symbol", "date"])
    logger.info(f"[S4] Loaded features: {len(result)} rows from {len(frames)} days")
    return result


def load_regime_map(start_date: str, end_date: str) -> dict:
    regime_map = {}
    current = pd.Timestamp(start_date)
    end     = pd.Timestamp(end_date)

    while current <= end:
        path = PROC_DIR / f"regime_{current.strftime('%Y-%m-%d')}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            raw = data.get("raw_regime", -1)
            regime_map[current.strftime("%Y-%m-%d")] = raw
        current += timedelta(days=1)

    logger.info(f"[S4] Loaded {len(regime_map)} regime labels")
    return regime_map


# Training

def train_xgb_regime(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    regime_name: str,
) -> tuple:
    if len(X_train) < 50:
        logger.warning(
            f"[S4] Insufficient training data for {regime_name}: {len(X_train)} samples"
        )
        return None, None, None

    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds     = model.predict_proba(X_val)[:, 1]
    val_binary    = (val_preds >= 0.5).astype(int)
    val_accuracy  = accuracy_score(y_val, val_binary)

    try:
        val_auc = roc_auc_score(y_val, val_preds)
    except Exception:
        val_auc = 0.5

    logger.info(
        f"[S4] {regime_name}: "
        f"acc={val_accuracy:.4f}, auc={val_auc:.4f}, "
        f"n_train={len(X_train)}, n_val={len(X_val)}"
    )

    return model, val_preds, val_accuracy


def run_walk_forward_training(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    regime_map:  dict,
    splits:      list,
) -> dict:
    from satellites.s4_forecaster.walk_forward import load_split_data

    oof_records = []
    regime_models = {r: [] for r in REGIME_LABELS.values()}
    feature_cols  = None

    for split in splits:
        logger.info(
            f"[S4] Split {split.split_id}: "
            f"train {split.train_start}→{split.train_end} | "
            f"val {split.val_start}→{split.val_end}"
        )

        result = load_split_data(
            split, features_df, targets_df, regime_map
        )
        X_train, y_train, X_val, y_val, r_train, r_val, feature_cols = result

        if len(X_train) == 0 or len(X_val) == 0:
            logger.warning(f"[S4] Split {split.split_id} skipped — insufficient data")
            continue

        # Train per-regime models
        for regime_id, regime_name in REGIME_LABELS.items():
            train_mask = (r_train == regime_id)
            val_mask   = (r_val   == regime_id)

            if train_mask.sum() < 50 or val_mask.sum() < 10:
                continue

            model, preds, acc = train_xgb_regime(
                X_train[train_mask], y_train[train_mask],
                X_val[val_mask],     y_val[val_mask],
                regime_name,
            )

            if model is not None:
                regime_models[regime_name].append({
                    "split_id": split.split_id,
                    "model":    model,
                    "accuracy": acc,
                })

                # Collect OOF predictions
                val_indices = np.where(val_mask)[0]
                for i, pred in zip(val_indices, preds):
                    oof_records.append({
                        "split_id":    split.split_id,
                        "regime":      regime_name,
                        "xgb_pred":    pred,
                        "actual":      y_val[i],
                    })

    oof_df = pd.DataFrame(oof_records)

    if not oof_df.empty:
        overall_acc = accuracy_score(
            oof_df["actual"],
            (oof_df["xgb_pred"] >= 0.5).astype(int)
        )
        logger.success(f"[S4] Walk-forward complete. OOF accuracy: {overall_acc:.4f}")

        # Per-regime accuracy
        for regime in REGIME_LABELS.values():
            r_df = oof_df[oof_df["regime"] == regime]
            if len(r_df) > 0:
                r_acc = accuracy_score(
                    r_df["actual"],
                    (r_df["xgb_pred"] >= 0.5).astype(int)
                )
                logger.info(f"[S4] {regime} OOF accuracy: {r_acc:.4f} (n={len(r_df)})")

    return {
        "regime_models": regime_models,
        "oof_df":        oof_df,
        "feature_cols":  feature_cols,
    }


def save_best_models(regime_models: dict, feature_cols: list):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for regime_name, model_list in regime_models.items():
        if not model_list:
            logger.warning(f"[S4] No models trained for {regime_name}")
            continue

        # Take the most recent split's model
        best = max(model_list, key=lambda x: x["accuracy"])
        model_path = MODEL_DIR / f"xgb_{regime_name.lower()}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(best["model"], f)

        logger.success(
            f"[S4] Saved {regime_name} model "
            f"(split {best['split_id']}, acc={best['accuracy']:.4f})"
        )

    # Save feature column order
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    logger.info(f"[S4] Feature columns saved: {len(feature_cols)} features")

def train_crisis_fallback(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    regime_map:  dict,
):
    """
    Trains a dedicated CRISIS model using high-volatility periods.
    Uses tighter features focused on risk-off behavior.
    """
    logger.info("[S4] Training CRISIS fallback model")

    # Merge data
    merged = features_df.merge(targets_df, on=["symbol", "date"], how="inner")
    merged["date_ts"] = pd.to_datetime(merged["date"])
    merged["regime"]  = merged["date_ts"].map(
        lambda d: regime_map.get(d.strftime("%Y-%m-%d"), -1)
    )

    # For crisis model: use high-volatility samples regardless of regime label
    # High vol = atr_14_norm in top 30% + negative recent returns
    if "atr_14_norm" in merged.columns and "ret_5d" in merged.columns:
        high_vol_mask = (
            (merged["atr_14_norm"] >= merged["atr_14_norm"].quantile(0.70)) |
            (merged["ret_5d"] <= merged["ret_5d"].quantile(0.20))
        )
        crisis_data = merged[high_vol_mask]
    else:
        crisis_data = merged

    logger.info(f"[S4] Crisis training samples: {len(crisis_data)}")

    if len(crisis_data) < 100:
        logger.warning("[S4] Insufficient crisis data — skipping")
        return

    feature_cols = [
        c for c in features_df.columns
        if c not in ["symbol", "date", "close"]
    ]

    X = crisis_data[feature_cols].fillna(0).values
    y = crisis_data["target"].values

    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Crisis-specific XGB: more conservative, stronger regularization
    crisis_params = {**XGB_PARAMS}
    crisis_params["max_depth"]         = 3   # Shallower
    crisis_params["learning_rate"]     = 0.01
    crisis_params["min_child_weight"]  = 50  # Conservative
    crisis_params["reg_lambda"]        = 3.0 # Stronger regularization

    model = xgb.XGBClassifier(**crisis_params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_preds  = model.predict_proba(X_val)[:, 1]
    val_binary = (val_preds >= 0.5).astype(int)
    acc = accuracy_score(y_val, val_binary)

    logger.success(f"[S4] CRISIS model trained: acc={acc:.4f} (n={len(X_train)})")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "xgb_crisis.pkl", "wb") as f:
        pickle.dump(model, f)

def print_feature_importance(regime: str = "ranging"):
    """Shows which features the model actually uses."""
    import json
    import pickle
    import pandas as pd
    from pathlib import Path

    model_path = Path("models/incumbent") / f"xgb_{regime}.pkl"
    feat_path  = Path("models/incumbent") / "feature_cols.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path) as f:
        features = json.load(f)

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(f"\nTop 15 features ({regime.upper()} model):")
    print("=" * 40)
    print(importance.head(15).to_string())
    print()
    print("Bottom 5 features (likely noise):")
    print(importance.tail(5).to_string())

# Inference
def predict_today(
    features_df:  pd.DataFrame,
    regime_label: str,
) -> pd.DataFrame:
    """
    Loads the appropriate regime model and predicts on today's features.
    Uses relative ranking to split LONG/SHORT — avoids all-one-direction bias.
    """
    model_path = MODEL_DIR / f"xgb_{regime_label.lower()}.pkl"

    # Fallback chain: RANGING → TRENDING → any available
    if not model_path.exists():
        for fallback in ["ranging", "trending", "crisis"]:
            fallback_path = MODEL_DIR / f"xgb_{fallback}.pkl"
            if fallback_path.exists():
                logger.warning(
                    f"[S4] No model for {regime_label}, "
                    f"falling back to {fallback}"
                )
                model_path = fallback_path
                break

    if not model_path.exists():
        logger.error("[S4] No XGBoost models found at all")
        return pd.DataFrame()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Align features
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].fillna(0).values
    preds = model.predict_proba(X)[:, 1]

    result = features_df[["symbol"]].copy()
    result["xgb_pred"] = preds

    # Use RELATIVE ranking to determine direction
    # Top 40% → LONG, Bottom 40% → SHORT, Middle 20% → skip
    upper_threshold = result["xgb_pred"].quantile(0.60)
    lower_threshold = result["xgb_pred"].quantile(0.40)

    result["direction"] = "NEUTRAL"
    result.loc[result["xgb_pred"] >= upper_threshold, "direction"] = "LONG"
    result.loc[result["xgb_pred"] <= lower_threshold, "direction"] = "SHORT"

    # Confidence = distance from median (0=uncertain, 1=very certain)
    median_pred = result["xgb_pred"].median()
    result["confidence"] = (result["xgb_pred"] - median_pred).abs() / 0.5
    result["confidence"] = result["confidence"].clip(0, 1)

    # Remove neutral signals
    result = result[result["direction"] != "NEUTRAL"]
    result = result.sort_values("xgb_pred", ascending=False)

    logger.debug(
        f"[S4] Predictions: {len(features_df)} stocks → "
        f"{(result['direction']=='LONG').sum()} LONG, "
        f"{(result['direction']=='SHORT').sum()} SHORT | "
        f"regime={regime_label}"
    )
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from satellites.s4_forecaster.walk_forward import generate_splits
    from satellites.s4_forecaster.target_builder import build_target_series
    from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols

    symbols = [s.replace(".NS", "") for s in get_nse_universe_symbols()]

    logger.info("[S4] Loading features for walk-forward training...")

    train_end = date.today() - timedelta(days=10)

    features_df = load_all_features(
        start_date=date(2022, 1, 1),
        end_date=train_end,   
    )

    if features_df.empty:
        logger.error("No features found. Run the pipeline for historical dates first.")
        sys.exit(1)

    logger.info("[S4] Building targets...")

    train_end = date.today() - timedelta(days=10)

    targets_df = build_target_series(
        symbols    = symbols,
        start_date = date(2022, 1, 1),
        end_date   = train_end,
    )

    logger.info("[S4] Loading regime map...")
    regime_map = load_regime_map(
        "2022-01-01",
        train_end.strftime("%Y-%m-%d"),  
    )

    splits = generate_splits(
        start_date=date(2022, 1, 1),
        end_date=date.today() - timedelta(days=10), 
        train_months=18,
        val_months=3,
        embargo_days=5,
    )

    results = run_walk_forward_training(
        features_df, targets_df, regime_map, splits
    )

    save_best_models(results["regime_models"], results["feature_cols"])

    logger.info("[S4] Training CRISIS fallback model")
    train_crisis_fallback(features_df, targets_df, regime_map)

    # Add to the bottom of xgb_model.py __main__ block
if __name__ == "__main__":
    print_feature_importance("ranging")
    print_feature_importance("trending")
    print_feature_importance("crisis")