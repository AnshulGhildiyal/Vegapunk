import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
from satellites.s3_features.feature_engineer import build_feature_matrix

UNIVERSE_SYMBOLS = None  # Will load from latest universe

def backfill_features(start_date: date, end_date: date):
    """Generates feature parquets for all trading days in range."""

    # Load latest universe as symbol reference
    universe_files = sorted(Path("data/universes").glob("universe_*.parquet"))
    if not universe_files:
        logger.error("No universe files found")
        return
    universe_df = pd.read_parquet(universe_files[-1])
    logger.info(f"[BACKFILL] Universe: {len(universe_df)} symbols")

    current = start_date
    saved = 0
    skipped = 0

    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        out_path = Path("data/processed") / f"features_{current.isoformat()}.parquet"
        if out_path.exists():
            skipped += 1
            current += timedelta(days=1)
            continue

        result = build_feature_matrix(universe_df, current)
        if result is not None:
            saved += 1
            logger.info(f"[BACKFILL] {current} → {result['n_stocks']} stocks")
        else:
            logger.warning(f"[BACKFILL] {current} → no data (holiday?)")

        current += timedelta(days=1)

    logger.success(f"[BACKFILL] Done. Saved: {saved}, Skipped: {skipped}")

if __name__ == "__main__":
    backfill_features(
        start_date=date(2022, 1, 1),
        end_date=date(2025, 12, 31),
    )