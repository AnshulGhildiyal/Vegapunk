import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from loguru import logger
import yaml
import sys

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = Path(CONFIG["paths"]["data_raw"]) / "bhavcopy"


sys.path.insert(0, ".")
from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols, _local_path


def download_year_chunk(symbols: list[str], start: str, end: str) -> int:
    """
    Downloads one chunk (e.g. one year) and saves per-day parquet files.
    Returns count of trading days saved.
    """
    logger.info(f"[HIST] Downloading {start} → {end} ({len(symbols)} symbols)")

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
        logger.error(f"[HIST] Download failed for {start}→{end}: {e}")
        return 0

    if raw.empty:
        logger.warning(f"[HIST] Empty response for {start}→{end}")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for dt in raw.index.normalize().unique():
        dt_date = dt.date()
        local = _local_path(dt_date)

        if local.exists():
            continue  # Already cached from previous run

        records = []
        for sym in symbols:
            try:
                sym_data = raw[sym]
                if dt not in sym_data.index:
                    continue
                row = sym_data.loc[dt]
                close = row.get("Close", np.nan)
                if pd.isna(close):
                    continue
                volume = int(row["Volume"]) if not pd.isna(row.get("Volume", np.nan)) else 0
                close = float(close)
                records.append({
                    "symbol":          sym.replace(".NS", ""),
                    "open":            round(float(row["Open"]), 2),
                    "high":            round(float(row["High"]), 2),
                    "low":             round(float(row["Low"]), 2),
                    "close":           close,
                    "volume":          volume,
                    "traded_value_cr": round(close * volume / 1e7, 4),
                    "date":            dt_date.isoformat(),
                })
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df.to_parquet(local, index=False)
            saved += 1

    logger.success(f"[HIST] Saved {saved} new trading days for {start}→{end}")
    return saved


def download_full_history(
    start_year: int = 2020,
    end_date: date = None,
    chunk_months: int = 6,
):
    """
    Downloads full history in chunks to avoid yfinance timeouts.
    Default: 2020-01-01 to today, in 6-month chunks.
    """
    if end_date is None:
        end_date = date.today()

    symbols = get_nse_universe_symbols()
    logger.info(f"[HIST] Starting full history download: {start_year}-01-01 → {end_date}")
    logger.info(f"[HIST] {len(symbols)} symbols, {chunk_months}-month chunks")


    chunks = []
    chunk_start = date(start_year, 1, 1)
    while chunk_start < end_date:
        chunk_end = min(
            chunk_start + relativedelta(months=chunk_months),
            end_date + timedelta(days=1)
        )
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end

    logger.info(f"[HIST] {len(chunks)} chunks to download")

    total_saved = 0
    for i, (start, end) in enumerate(chunks, 1):
        logger.info(f"[HIST] Chunk {i}/{len(chunks)}: {start} → {end}")
        saved = download_year_chunk(symbols, start, end)
        total_saved += saved

    all_cached = sorted(RAW_DIR.glob("bhavcopy_*.parquet"))
    logger.success(f"[HIST] Complete. {len(all_cached)} total trading days cached.")
    logger.info(f"[HIST] Date range: {all_cached[0].stem.split('_')[1]} → {all_cached[-1].stem.split('_')[1]}")


def verify_cache():
    """Quick check of what's in cache."""
    files = sorted(RAW_DIR.glob("bhavcopy_*.parquet"))
    if not files:
        logger.error("[HIST] Cache is empty!")
        return

    dates = [f.stem.replace("bhavcopy_", "") for f in files]

    sample = pd.read_parquet(files[-1])
    
    print(f"\n{'='*50}")
    print(f"CACHE SUMMARY")
    print(f"{'='*50}")
    print(f"Total trading days : {len(files)}")
    print(f"First date         : {dates[0]}")
    print(f"Last date          : {dates[-1]}")
    print(f"Latest file stocks : {len(sample)}")
    print(f"Storage used       : {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VEGAPUNK Historical Data Downloader")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--verify", action="store_true", help="Just verify cache, no download")
    args = parser.parse_args()

    if args.verify:
        verify_cache()
    else:
        download_full_history(start_year=args.start_year)
        verify_cache()