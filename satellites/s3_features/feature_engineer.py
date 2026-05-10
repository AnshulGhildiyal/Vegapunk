import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml

from satellites.s2_sentiment.sentiment_engine import build_sentiment_features

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED_DIR = Path(CONFIG["paths"]["data_processed"])
RAW_DIR = Path(CONFIG["paths"]["data_raw"]) / "bhavcopy"


def load_price_history(symbols: list[str], end_date: date, lookback_days: int = 120) -> pd.DataFrame:
    frames = []
    for i in range(lookback_days):
        dt = end_date - timedelta(days=i)
        local = RAW_DIR / f"bhavcopy_{dt.isoformat()}.parquet"
        if not local.exists():
            continue
        df = pd.read_parquet(local)
        df = df[df["symbol"].isin(symbols)]
        frames.append(df)

    if not frames:
        logger.error(f"[S3] No cached data found for lookback ending {end_date}")
        return pd.DataFrame()

    history = (pd.concat(frames).drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True))

    logger.info(f"[S3] Loaded {len(frames)} days × {history['symbol'].nunique()} symbols")
    return history


# Feature Family 1: Price Returns
def compute_price_features(history: pd.DataFrame) -> pd.DataFrame:
    results = []

    for symbol, grp in history.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < 22:  # Need at least 22 days
            continue

        latest = grp.iloc[-1]
        closes = grp["close"].values
        n = len(closes)

        def log_ret(periods):
            if n <= periods:
                return np.nan
            return np.log(closes[-1] / closes[-1 - periods])

        high_52w = grp["high"].max()
        dist_52w_high = (high_52w - closes[-1]) / high_52w if high_52w > 0 else np.nan

        # Gap open: today's open vs yesterday's close
        gap_open = np.nan
        if n >= 2:
            gap_open = (grp["open"].iloc[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else np.nan

        results.append({
            "symbol":          symbol,
            "date":            latest["date"],
            "close":           closes[-1],
            # Return features
            "ret_1d":          log_ret(1),
            "ret_5d":          log_ret(5),
            "ret_10d":         log_ret(10),
            "ret_20d":         log_ret(20),
            # Derived
            "ret_1d_sq":       log_ret(1) ** 2 if not np.isnan(log_ret(1)) else np.nan,
            "high_low_ratio":  (latest["high"] - latest["low"]) / latest["close"] if latest["close"] > 0 else np.nan,
            "gap_open":        gap_open,
            "dist_52w_high":   dist_52w_high,
        })

    df = pd.DataFrame(results)
    logger.info(f"[S3] Price features: {len(df)} stocks × {len(df.columns)-3} features")
    return df


# Feature Family 2: Volume Features 

def compute_volume_features(history: pd.DataFrame) -> pd.DataFrame:
    results = []

    for symbol, grp in history.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < 22:
            continue

        volumes = grp["volume"].values
        closes  = grp["close"].values
        n = len(volumes)


        avg_vol_5d  = volumes[-6:-1].mean() if n >= 6 else np.nan   # Exclude today
        avg_vol_20d = volumes[-21:-1].mean() if n >= 21 else np.nan

        today_vol = volumes[-1]

        vol_ratio_5d  = today_vol / avg_vol_5d  if avg_vol_5d  > 0 else np.nan
        vol_ratio_20d = today_vol / avg_vol_20d if avg_vol_20d > 0 else np.nan

        obv_slope = np.nan
        if n >= 20:
            obv = []
            obv_val = 0
            for i in range(1, n):
                if closes[i] > closes[i-1]:
                    obv_val += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv_val -= volumes[i]
                obv.append(obv_val)
            if len(obv) >= 20:
                obv_20 = np.array(obv[-20:])
                x = np.arange(20)
                obv_slope = float(np.polyfit(x, obv_20, 1)[0])
                if avg_vol_20d > 0:
                    obv_slope = obv_slope / avg_vol_20d

        pv_corr = np.nan
        if n >= 20:
            p20 = closes[-20:]
            v20 = volumes[-20:]
            if v20.std() > 0 and p20.std() > 0:
                pv_corr = float(np.corrcoef(p20, v20)[0, 1])

        results.append({
            "symbol":        symbol,
            "vol_ratio_5d":  vol_ratio_5d,
            "vol_ratio_20d": vol_ratio_20d,
            "obv_slope":     obv_slope,
            "pv_corr_20d":   pv_corr,
        })

    df = pd.DataFrame(results)
    logger.info(f"[S3] Volume features: {len(df)} stocks × {len(df.columns)-1} features")
    return df


# Feature Family 3: Technical Indicators

def compute_technical_features(history: pd.DataFrame) -> pd.DataFrame:
    results = []

    for symbol, grp in history.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < 26:
            continue

        closes = grp["close"].values
        highs  = grp["high"].values
        lows   = grp["low"].values
        n = len(closes)

        rsi_14 = np.nan
        if n >= 15:
            deltas = np.diff(closes[-15:])
            gains  = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_14 = 100 - (100 / (1 + rs))
            else:
                rsi_14 = 100.0

        macd_hist = np.nan
        if n >= 26:
            def ema(arr, span):
                s = pd.Series(arr)
                return s.ewm(span=span, adjust=False).mean().values

            ema12 = ema(closes, 12)
            ema26 = ema(closes, 26)
            macd_line   = ema12 - ema26
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            macd_hist   = float(macd_line[-1] - signal_line[-1])
            if closes[-1] > 0:
                macd_hist = macd_hist / closes[-1]

        # Bollinger Band position (20, 2σ)
        bb_position = np.nan
        if n >= 20:
            c20   = closes[-20:]
            bb_mid = c20.mean()
            bb_std = c20.std()
            if bb_std > 0:
                bb_upper = bb_mid + 2 * bb_std
                bb_lower = bb_mid - 2 * bb_std
                bb_range = bb_upper - bb_lower
                bb_position = (closes[-1] - bb_lower) / bb_range if bb_range > 0 else 0.5

        # ATR(14) normalized
        atr_14_norm = np.nan
        if n >= 15:
            trs = []
            for i in range(-14, 0):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1])
                )
                trs.append(tr)
            atr = np.mean(trs)
            atr_14_norm = atr / closes[-1] if closes[-1] > 0 else np.nan

        # MA cross signal: sign of (EMA20 - EMA50)
        ma_cross = np.nan
        if n >= 50:
            s = pd.Series(closes)
            ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
            ma_cross = np.sign(ema20 - ema50)

        results.append({
            "symbol":      symbol,
            "rsi_14":      rsi_14,
            "macd_hist":   macd_hist,
            "bb_position": bb_position,
            "atr_14_norm": atr_14_norm,
            "ma_cross":    ma_cross,
        })

    df = pd.DataFrame(results)
    logger.info(f"[S3] Technical features: {len(df)} stocks × {len(df.columns)-1} features")
    return df


# Cross-Sectional Features

def compute_cross_sectional_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    n  = len(df)

    # Percentile rank of 20d return within universe
    df["cs_mom_rank_20d"] = df["ret_20d"].rank(pct=True)
    df["cs_mom_rank_5d"]  = df["ret_5d"].rank(pct=True)

    # Demeaned returns (stock return - universe median)
    med_ret_5d  = df["ret_5d"].median()
    med_ret_20d = df["ret_20d"].median()
    df["excess_ret_5d"]  = df["ret_5d"]  - med_ret_5d
    df["excess_ret_20d"] = df["ret_20d"] - med_ret_20d

    # Volume rank within universe
    df["cs_vol_rank"] = df.get("vol_ratio_5d", pd.Series(np.nan, index=df.index)).rank(pct=True)

    logger.info(f"[S3] Cross-sectional features: {n} stocks")
    return df


# Main Builder

def build_feature_matrix(universe_df: pd.DataFrame, run_date: date) -> dict | None:
    logger.info(f"[S3] Building feature matrix for {run_date} ({len(universe_df)} stocks)")

    symbols = universe_df["symbol"].tolist()
    history = load_price_history(symbols, run_date, lookback_days=120)

    if history.empty:
        logger.error("[S3] No history loaded — cannot build features")
        return None

    # Compute each family
    price_feats  = compute_price_features(history)
    volume_feats = compute_volume_features(history)
    tech_feats   = compute_technical_features(history)

    if price_feats.empty:
        logger.error("[S3] Price features empty")
        return None

    full = (price_feats
            .merge(volume_feats, on="symbol", how="left")
            .merge(tech_feats,   on="symbol", how="left"))

    full = compute_cross_sectional_features(full)

    # Sentiment placeholder (zeros until S2 is live)
    try:
        sentiment_df = build_sentiment_features(run_date, symbols)
        full = full.merge(
            sentiment_df[[
                "symbol", "raw_sentiment", "sentiment_momentum",
                "sentiment_volume", "sentiment_volatility"
            ]],
            on="symbol", how="left"
        )
        for col in ["raw_sentiment", "sentiment_momentum",
                    "sentiment_volume", "sentiment_volatility"]:
            full[col] = full[col].fillna(0.0)

        # Price-sentiment divergence: price went up but sentiment is negative = potential reversal
        full["price_sentiment_divergence"] = (
            full["ret_5d"].fillna(0) - full["raw_sentiment"].fillna(0)
        )
        logger.info(
            f"[S3] Sentiment wired: "
            f"{(full['raw_sentiment'] != 0).sum()} stocks with signals"
        )
    except Exception as e:
        logger.warning(f"[S3] Sentiment failed: {e} — using zeros")
        for col in ["raw_sentiment", "sentiment_momentum", "sentiment_volume",
                    "sentiment_volatility", "price_sentiment_divergence"]:
            full[col] = 0.0

    # Drop rows with too many NaNs
    feature_cols = [c for c in full.columns if c not in ["symbol", "date", "close"]]
    nan_threshold = 0.5
    full = full[full[feature_cols].isna().mean(axis=1) < nan_threshold]

    logger.info(f"[S3] Feature matrix: {len(full)} stocks × {len(feature_cols)} features")

    # Save to parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"features_{run_date.isoformat()}.parquet"
    full.to_parquet(out_path, index=False)
    logger.success(f"[S3] Saved → {out_path}")

    return {
        "xgb":      full,
        "features": feature_cols,
        "n_stocks": len(full),
        "n_features": len(feature_cols),
        "path": str(out_path),
    }


if __name__ == "__main__":
    from datetime import date
    import sys
    sys.path.insert(0, ".")

    # Load last universe snapshot
    from pathlib import Path
    universe_files = sorted(Path("data/universes").glob("universe_*.parquet"))
    if not universe_files:
        print("No universe files found. Run universe_builder.py first.")
        sys.exit(1)

    latest_universe = pd.read_parquet(universe_files[-1])
    run_date = date(2026, 4, 28)

    result = build_feature_matrix(latest_universe, run_date)

    if result:
        df = result["xgb"]
        print(f"\n Feature matrix built:")
        print(f"   Stocks  : {result['n_stocks']}")
        print(f"   Features: {result['n_features']}")
        print(f"\nSample (first 5 stocks, key features):")
        cols = ["symbol", "ret_1d", "ret_5d", "ret_20d",
                "vol_ratio_5d", "rsi_14", "bb_position", "cs_mom_rank_20d"]
        print(df[cols].head().to_string(index=False))
        print(f"\nNaN summary:")
        nan_pct = df[result["features"]].isna().mean() * 100
        print(nan_pct[nan_pct > 0].sort_values(ascending=False).head(10))