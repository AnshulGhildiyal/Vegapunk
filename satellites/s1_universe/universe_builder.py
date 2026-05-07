"""
S1 — Universe Builder
Runs the 5-gate filter pipeline on Bhavcopy data.
Outputs a dated universe snapshot Parquet file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml

from satellites.s1_universe.bhavcopy_downloader import (
    load_bhavcopy,
    download_bulk_history,
    _local_path,
)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S1_CFG = CONFIG["s1"]
UNIVERSE_DIR = Path(CONFIG["paths"]["data_universes"])
RAW_DIR = Path(CONFIG["paths"]["data_raw"])


# ── Gate 1: Liquidity ──────────────────────────────────────────────────────────

def gate_liquidity(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Requires 60-day average daily traded value >= min_adtv_crore
    and average daily volume >= min_daily_volume.
    Uses rolling 60-day history if available, else uses today's single-day values.
    """
    min_adtv = S1_CFG["min_adtv_crore"]
    min_vol  = S1_CFG["min_daily_volume"]

    if history is not None and len(history) > 5:
        adtv = (history.groupby("symbol")["traded_value_cr"]
                .mean()
                .rename("adtv_60d"))
        avg_vol = (history.groupby("symbol")["volume"].mean().rename("avg_vol_60d"))
        df = df.merge(adtv, on="symbol", how="left")
        df = df.merge(avg_vol, on="symbol", how="left")
    else:
        # Fallback: use today's single-day value as proxy
        df["adtv_60d"]  = df["traded_value_cr"]
        df["avg_vol_60d"] = df["volume"]
        logger.warning("[S1] Gate 1: No 60d history — using single-day values as proxy")

    before = len(df)
    df = df[(df["adtv_60d"] >= min_adtv) & (df["avg_vol_60d"] >= min_vol)]
    logger.info(f"[S1] Gate 1 (Liquidity): {before} → {len(df)} stocks")
    return df


# ── Gate 2: Price Filter ───────────────────────────────────────────────────────

def gate_price(df: pd.DataFrame) -> pd.DataFrame:
    min_p = S1_CFG["min_price"]
    max_p = S1_CFG["max_price"]
    before = len(df)
    df = df[(df["close"] >= min_p) & (df["close"] <= max_p)]
    logger.info(f"[S1] Gate 2 (Price ₹{min_p}–₹{max_p}): {before} → {len(df)} stocks")
    return df


# ── Gate 3: Listing Age ────────────────────────────────────────────────────────

def load_listing_dates() -> pd.DataFrame | None:
    """
    Loads NSE listing dates from a local reference file.
    File: data/raw/nse_symbol_master.csv
    Download once from: https://nseindia.com/market-data/securities-available-for-trading
    """
    master_path = RAW_DIR / "nse_symbol_master.csv"
    if not master_path.exists():
        logger.warning("[S1] Gate 3: nse_symbol_master.csv not found — skipping listing age filter")
        return None

    df = pd.read_csv(master_path)
    df.columns = df.columns.str.strip().str.upper()

    # NSE master has columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, ...
    df = df.rename(columns={
        "SYMBOL": "symbol",
        "DATE OF LISTING": "listing_date"
    })
    df["listing_date"] = pd.to_datetime(df["listing_date"], dayfirst=True, errors="coerce")
    return df[["symbol", "listing_date"]].dropna()


def gate_listing_age(df: pd.DataFrame, run_date: date) -> pd.DataFrame:
    listing = load_listing_dates()
    if listing is None:
        logger.warning("[S1] Gate 3: Skipped (no listing date data)")
        return df

    min_years = S1_CFG["min_listing_years"]
    cutoff = pd.Timestamp(run_date) - pd.DateOffset(years=min_years)

    df = df.merge(listing, on="symbol", how="left")
    before = len(df)
    # Stocks without listing date: keep them (benefit of the doubt)
    df = df[(df["listing_date"].isna()) | (df["listing_date"] <= cutoff)]
    logger.info(f"[S1] Gate 3 (Listing ≥{min_years}y): {before} → {len(df)} stocks")
    return df


# ── Gate 4: Surveillance Exclusion ────────────────────────────────────────────

def load_asm_gsm_list() -> set:
    """
    Returns set of symbols currently on NSE's ASM or GSM surveillance lists.
    These are downloaded manually and stored locally for now.
    Future: automate scraping from NSE website.
    """
    asm_path = RAW_DIR / "asm_list.csv"
    gsm_path = RAW_DIR / "gsm_list.csv"

    symbols = set()
    for path in [asm_path, gsm_path]:
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.upper()
            if "SYMBOL" in df.columns:
                symbols.update(df["SYMBOL"].str.strip().tolist())

    if not symbols:
        logger.warning("[S1] Gate 4: No ASM/GSM files found — surveillance filter skipped")

    return symbols


def gate_surveillance(df: pd.DataFrame) -> pd.DataFrame:
    excluded = load_asm_gsm_list()
    if not excluded:
        return df
    before = len(df)
    df = df[~df["symbol"].isin(excluded)]
    logger.info(f"[S1] Gate 4 (Surveillance): {before} → {len(df)} stocks "
                f"(removed {before - len(df)})")
    return df


# ── Gate 5: Sector Diversification Cap ────────────────────────────────────────

def load_sector_map() -> pd.DataFrame | None:
    """
    Loads NSE industry/sector classification.
    File: data/raw/nse_sector_map.csv  (columns: symbol, sector)
    """
    sector_path = RAW_DIR / "nse_sector_map.csv"
    if not sector_path.exists():
        logger.warning("[S1] Gate 5: nse_sector_map.csv not found — sector cap skipped")
        return None

    df = pd.read_csv(sector_path)
    df.columns = df.columns.str.strip().str.upper()
    return df.rename(columns={"SYMBOL": "symbol", "SECTOR": "sector"})[["symbol", "sector"]]


def gate_sector_cap(df: pd.DataFrame) -> pd.DataFrame:
    sector_map = load_sector_map()
    if sector_map is None:
        df["sector"] = "UNKNOWN"
        return df

    df = df.merge(sector_map, on="symbol", how="left")
    df["sector"] = df["sector"].fillna("UNKNOWN")

    max_pct = S1_CFG["max_sector_pct"]
    max_stocks = int(len(df) * max_pct)

    # Sort by liquidity descending within each sector, keep top N per sector
    df = df.sort_values("adtv_60d", ascending=False)
    capped = (df.groupby("sector").head(max_stocks).reset_index(drop=True))

    before = len(df)
    logger.info(f"[S1] Gate 5 (Sector cap {max_pct*100:.0f}%): {before} → {len(capped)} stocks")
    return capped

# Get last trading day

def get_last_trading_day(from_date: date, max_lookback: int = 15) -> date | None:
    for i in range(max_lookback):
        candidate = from_date - timedelta(days=i)
        if candidate.weekday() >= 5:  # Skip weekends
            continue
        local = _local_path(candidate)
        if local.exists():
            return candidate
        # Try fetching
        df = load_bhavcopy(candidate)
        if df is not None and len(df) > 10:
            return candidate
    logger.error(f"[S1] No trading day found in last {max_lookback} days")
    return None

# Main Builder 

def build_universe(run_date: date) -> pd.DataFrame | None:
    logger.info(f"[S1] Building universe for {run_date}")

    # Check cache freshness before downloading
    recent_cached = sorted(RAW_DIR.glob("bhavcopy_*.parquet"))
    last_cached_date = None
    if recent_cached:
        last_cached_date = date.fromisoformat(
            recent_cached[-1].stem.replace("bhavcopy_", "")
        )

    if last_cached_date is None or (run_date - last_cached_date).days > 1:
        download_bulk_history(end_date=run_date, days=90)
    else:
        logger.info(f"[S1] Cache fresh (last: {last_cached_date}) — skipping download")

    trading_day = get_last_trading_day(run_date)
    if trading_day is None:
        logger.error("[S1] Could not find a valid trading day")
        return None

    today_df = load_bhavcopy(trading_day)
    if today_df is None:
        return None

    history_frames = []
    for i in range(1, 61):
        hist_date = trading_day - timedelta(days=i)
        local = _local_path(hist_date)
        if local.exists():
            history_frames.append(pd.read_parquet(local))

    history = pd.concat(history_frames) if history_frames else None
    logger.info(f"[S1] Loaded {len(history_frames)} cached days for ADTV")

    universe = (today_df
                .pipe(gate_liquidity, history)
                .pipe(gate_price)
                .pipe(lambda df: gate_listing_age(df, trading_day))
                .pipe(gate_surveillance)
                .pipe(gate_sector_cap))

    universe["run_date"]    = trading_day.isoformat()
    universe["s1_approved"] = True

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UNIVERSE_DIR / f"universe_{trading_day.isoformat()}.parquet"
    universe.to_parquet(out_path, index=False)
    logger.success(f"[S1] Universe saved: {len(universe)} stocks → {out_path}")
    logger.info(f"[S1] Sector breakdown:\n{universe['sector'].value_counts().to_string()}")

    return universe


if __name__ == "__main__":
    from datetime import date
    result = build_universe(date.today())
    if result is not None:
        print(f"\n Universe: {len(result)} stocks approved")
        print(result[["symbol", "close", "adtv_60d", "sector"]].head(20).to_string())