from pandas import unique
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = Path(CONFIG["paths"]["data_raw"]) / "bhavcopy"


def _local_path(dt: date) -> Path:
    return RAW_DIR / f"bhavcopy_{dt.isoformat()}.parquet"


def get_nse_universe_symbols() -> list[str]:
    nifty500 = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "BAJFINANCE", "WIPRO", "ULTRACEMCO", "NESTLEIND", "ONGC",
        "NTPC", "POWERGRID", "TECHM", "SUNPHARMA", "HCLTECH",
        "TATASTEEL", "ADANIENT", "ADANIPORTS", "BAJAJFINSV",
        "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "GRASIM",
        "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "JSWSTEEL", "M&M",
        "BRITANNIA", "COALINDIA", "APOLLOHOSP", "BPCL", "SHREECEM",
        "TATACONSUM", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO", "UPL",
        "PIDILITIND", "HAVELLS", "BERGEPAINT", "MARICO", "DABUR",
        "GODREJCP", "COLPAL", "PAGEIND", "BOSCHLTD", "SIEMENS",
        "ABB", "TORNTPHARM", "LUPIN", "BIOCON", "AUROPHARMA",
        "GLENMARK", "LALPATHLAB", "APOLLOTYRE", "MRF", "BALKRISIND",
        "CEATLTD", "HDFCAMC", "ICICIGI", "ICICIPRULI", "SBICARD",
        "CHOLAFIN", "M&MFIN", "LICHSGFIN", "RECLTD", "PFC",
        "IRFC", "HUDCO", "MANAPPURAM", "MUTHOOTFIN", "SHRIRAMFIN",
        "SUNDARMFIN", "INDIAMART", "NAUKRI", "POLICYBZR", "PAYTM",
        "NYKAA", "DELHIVERY", "IRCTC", "CONCOR",
        "TATAPOWER", "ADANIGREEN", "TORNTPOWER", "CESC", "JSWENERGY",
        "NHPC", "SJVN", "OBEROIRLTY", "DLF", "GODREJPROP",
        "PRESTIGE", "BRIGADE", "SOBHA", "PHOENIXLTD", "ZYDUSLIFE",
        "PFIZER", "GLAXO", "ALKEM", "GRANULES", "LAURUSLABS",
        "TATACHEM", "DEEPAKNTR", "NAVINFLUOR", "COROMANDEL", "GNFC",
        "PETRONET", "GAIL", "IOC", "BPCL",
        "VEDL", "HINDZINC", "NMDC", "SAIL", "NATIONALUM",
        "GUJGASLTD", "IGL", "MGL", "DIXON", "AMBER",
        "VOLTAS", "POLYCAB", "HAVELLS", "WHIRLPOOL", "SYMPHONY",
        "DMART", "TRENT", "TITAN", "BATAINDIA", "RELAXO",
        "MANYAVAR", "BAJAJHLDNG", "CANFINHOME", "AAVAS",
        "SUNPHARMA", "DIVISLAB", "CIPLA", "DRREDDY", "LUPIN",
        "FORTIS", "APOLLOHOSP", "MAXHEALTH", "NH", "THYROCARE",
        "SBILIFE", "HDFCLIFE", "ICICIPRULI", "MFSL",
        "HINDUNILVR", "NESTLEIND", "BRITANNIA", "COLPAL", "DABUR",
        "MARICO", "GODREJCP", "TATACONSUM", "ITC", "BRITANNIA",
    ]

    # Deduplicate
    seen = set()
    unique = []
    for s in nifty500:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    
    BLOCKLIST = {
    "TATAMOTORS", "ZOMATO", "LTFH", "NIPPONLIFE", "HPCL",
    "MCDOWELL-N", "CHAMBALFERT", "WELSPUNIND", "APCOTEX",
    "FINOLEX", "SKF", "VARDHMAN", "CPCL", "CADILAHC",
    "NOVARTIS", "ADANITRANS", "GMRINFRA", "AARTI",
    "SUPPETRO", "METRO", "VEDANT",
    }
    clean = [s for s in unique if s.replace(".NS", "") not in BLOCKLIST]
    return [f"{s}.NS" for s in clean]



def download_bulk_history(end_date: date, days: int = 90) -> dict[str, pd.DataFrame]:
    start = (end_date - timedelta(days=days)).isoformat()
    end = (end_date + timedelta(days=1)).isoformat()
    symbols = get_nse_universe_symbols()

    logger.info(f"[S1] Bulk download: {len(symbols)} symbols from {start} to {end_date}")

    try:
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=True,
            threads=True,
        )
    except Exception as e:
        logger.error(f"[S1] Bulk download failed: {e}")
        return {}

    if raw.empty:
        logger.error("[S1] Bulk download returned empty DataFrame")
        return {}
    daily_data = {}
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = raw.index.normalize().unique()

    for dt in all_dates:
        dt_date = dt.date()
        local = _local_path(dt_date)

        if local.exists():
            continue

        records = []
        for sym in symbols:
            try:
                sym_data = raw[sym]
                if dt not in sym_data.index:
                    continue
                row = sym_data.loc[dt]
                if pd.isna(row.get("Close", np.nan)):
                    continue
                close = float(row["Close"])
                volume = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
                records.append({
                    "symbol": sym.replace(".NS", ""),
                    "open":   round(float(row["Open"]), 2),
                    "high":   round(float(row["High"]), 2),
                    "low":    round(float(row["Low"]), 2),
                    "close":  close,
                    "volume": volume,
                    "traded_value_cr": round(close * volume / 1e7, 4),
                    "date":   dt_date.isoformat(),
                })
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df.to_parquet(local, index=False)
            daily_data[dt_date.isoformat()] = df

    cached = len(list(RAW_DIR.glob("bhavcopy_*.parquet")))
    logger.success(f"[S1] Bulk download complete. {cached} days now cached in {RAW_DIR}")
    return daily_data


def load_bhavcopy(dt: date) -> pd.DataFrame | None:
    """Load a single day. Uses cache; triggers bulk download if missing."""
    local = _local_path(dt)
    if local.exists():
        return pd.read_parquet(local)

    logger.info(f"[S1] {dt} not cached — running bulk download")
    download_bulk_history(end_date=dt, days=90)

    if local.exists():
        return pd.read_parquet(local)

    logger.warning(f"[S1] No data for {dt} after bulk download (holiday or weekend)")
    return None


if __name__ == "__main__":
    from datetime import date
    download_bulk_history(end_date=date(2026, 4, 28), days=90)
    df = load_bhavcopy(date(2026, 4, 28))
    if df is not None:
        print(df[["symbol", "close", "volume", "traded_value_cr"]].head(15))
        print(f"\nTotal: {len(df)} stocks")