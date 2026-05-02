import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import yaml

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = Path(CONFIG["paths"]["data_raw"]) / "bhavcopy"


def build_target_series(
    symbols: list[str],
    start_date: date,
    end_date: date,
    horizon: int = 5,
) -> pd.DataFrame:

    logger.info(
        f"[S4] Building targets: {len(symbols)} symbols, "
        f"{start_date} → {end_date}, horizon={horizon}d"
    )

    # Load all daily closes into a pivot table
    frames = []
    current = start_date
    # Load extra days beyond end_date for forward return calculation
    extended_end = end_date + timedelta(days=horizon * 3)

    while current <= extended_end:
        local = RAW_DIR / f"bhavcopy_{current.isoformat()}.parquet"
        if local.exists():
            df = pd.read_parquet(local)[["symbol", "date", "close"]]
            df = df[df["symbol"].isin(symbols)]
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        logger.error("[S4] No data found for target building")
        return pd.DataFrame()

    all_data = pd.concat(frames).drop_duplicates(subset=["symbol", "date"])
    all_data["date"] = pd.to_datetime(all_data["date"])

    # Pivot: rows=dates, cols=symbols
    pivot = all_data.pivot(index="date", columns="symbol", values="close")
    pivot = pivot.sort_index()

    # Forward log return
    fwd_ret = np.log(pivot.shift(-horizon) / pivot)

    # Convert back to long format
    records = []
    for sym in symbols:
        if sym not in fwd_ret.columns:
            continue
        sym_fwd = fwd_ret[sym].dropna()
        for dt, ret in sym_fwd.items():
            if pd.Timestamp(start_date) <= dt <= pd.Timestamp(end_date):
                records.append({
                    "symbol":     sym,
                    "date":       dt.date().isoformat(),
                    "fwd_ret_5d": round(float(ret), 6),
                    "target":     1 if ret > 0 else 0,
                })

    result = pd.DataFrame(records)
    logger.info(
        f"[S4] Targets built: {len(result)} samples, "
        f"positive rate: {result['target'].mean():.3f}"
    )
    return result


if __name__ == "__main__":
    from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols

    symbols = [s.replace(".NS", "") for s in get_nse_universe_symbols()]
    targets = build_target_series(
        symbols=symbols[:20],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    print(targets.head(20).to_string())
    print(f"\nPositive rate: {targets['target'].mean():.3f}")
    print(f"Total samples: {len(targets)}")