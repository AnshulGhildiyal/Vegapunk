import sys
sys.path.insert(0, ".")

from datetime import date, timedelta
from pathlib import Path
from loguru import logger
from satellites.s2_sentiment.sentiment_engine import build_sentiment_features
from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols

def backfill_sentiment(days: int = 30):
    symbols = [s.replace(".NS", "") for s in get_nse_universe_symbols()]
    today = date.today()

    saved = 0
    for i in range(days, 0, -1):
        target_date = today - timedelta(days=i)
        if target_date.weekday() >= 5:
            continue

        score_path = Path("data/raw/sentiment/scores") / f"scores_{target_date.isoformat()}.json"
        if score_path.exists():
            continue

        logger.info(f"[BACKFILL] Sentiment for {target_date}")
        try:
            build_sentiment_features(target_date, symbols)
            saved += 1
        except Exception as e:
            logger.warning(f"[BACKFILL] Failed {target_date}: {e}")

    logger.success(f"[BACKFILL] Sentiment backfill complete: {saved} days")

if __name__ == "__main__":
    backfill_sentiment(days=30)