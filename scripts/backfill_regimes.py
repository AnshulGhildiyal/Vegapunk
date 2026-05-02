import sys
sys.path.insert(0, ".")

from datetime import date, timedelta
from pathlib import Path
from loguru import logger
from satellites.s7_regime.regime_detector import detect_regime

def backfill_regimes(start_date: date, end_date: date):
    current = start_date
    saved = 0

    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        out_path = Path("data/processed") / f"regime_{current.isoformat()}.json"
        if out_path.exists():
            current += timedelta(days=1)
            continue

        result = detect_regime(current)
        if result.get("regime_label") != "UNKNOWN":
            saved += 1

        current += timedelta(days=1)

    logger.success(f"[BACKFILL] Regimes saved: {saved}")

if __name__ == "__main__":
    backfill_regimes(date(2022, 1, 1), date(2025, 12, 31))