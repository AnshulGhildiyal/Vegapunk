import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger

from satellites.s2_sentiment.nse_announcements import (
    fetch_nse_announcements,
    fetch_google_news,
)
from satellites.s2_sentiment.finbert_scorer import batch_score_symbols

SENTIMENT_DIR = Path("data/raw/sentiment/scores")
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


# Add module-level cache
_announcement_cache: dict[str, list[dict]] = {}

def get_all_headlines(
    symbol: str,
    end_date: date,
    lookback_days: int = 5,
) -> list[str]:
    headlines = []

    # NSE Announcements — use cached data
    for i in range(lookback_days):
        d = end_date - timedelta(days=i)
        date_str = d.isoformat()

        # Load once per date, cache in memory
        if date_str not in _announcement_cache:
            _announcement_cache[date_str] = fetch_nse_announcements(d)

        for ann in _announcement_cache[date_str]:
            if ann["symbol"] == symbol and ann["headline"]:
                headlines.append(f"[NSE] {ann['headline']}")

    # Google News (already fast)
    for h in fetch_google_news(symbol):
        headlines.append(f"[NEWS] {h}")

    return headlines


def build_sentiment_features(
    run_date: date,
    universe_symbols: list[str],
) -> pd.DataFrame:
    """
    Main S2 function called by S3 daily.
    Returns DataFrame with 5 sentiment features per symbol.
    """
    score_path = SENTIMENT_DIR / f"scores_{run_date.isoformat()}.json"

    # Use cached scores if available
    if score_path.exists():
        with open(score_path) as f:
            cached = json.load(f)
        logger.info(f"[S2] Cached scores loaded: {run_date}")
    else:
        # Pre-load NSE announcements for all lookback days into cache
        for i in range(5):
            d = run_date - timedelta(days=i)
            date_str = d.isoformat()
            if date_str not in _announcement_cache:
                _announcement_cache[date_str] = fetch_nse_announcements(d)

        # Only fetch Google News for stocks with NSE activity today
        symbols_with_nse = {
            ann["symbol"]
            for ann in _announcement_cache.get(run_date.isoformat(), [])
        }

        symbol_headlines = {}
        for sym in universe_symbols:
            nse_headlines = []
            for i in range(5):
                d = run_date - timedelta(days=i)
                for ann in _announcement_cache.get(d.isoformat(), []):
                    if ann["symbol"] == sym and ann["headline"]:
                        nse_headlines.append(f"[NSE] {ann['headline']}")

            google_headlines = []
            if sym in symbols_with_nse:
                google_headlines = [
                    f"[NEWS] {h}" for h in fetch_google_news(sym)
                ]

            all_headlines = nse_headlines + google_headlines
            if all_headlines:
                symbol_headlines[sym] = all_headlines

        logger.info(
            f"[S2] {len(symbol_headlines)}/{len(universe_symbols)} "
            f"symbols have news for {run_date} "
            f"({len(symbols_with_nse)} with NSE announcements)"
        )
        if symbol_headlines:
            cached = batch_score_symbols(symbol_headlines)
        else:
            cached = {}

        # Save
        with open(score_path, "w") as f:
            json.dump(cached, f, indent=2)

    # Build feature DataFrame
    records = []
    for sym in universe_symbols:
        today_score = cached.get(sym, {}).get("score", 0.0)

        # Load historical scores for momentum/volatility
        hist_scores = []
        for i in range(1, 11):
            hist_path = SENTIMENT_DIR / f"scores_{(run_date - timedelta(days=i)).isoformat()}.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                hist_scores.append(hist.get(sym, {}).get("score", 0.0))

        # Derived features
        avg_5d = np.mean(hist_scores[:5]) if len(hist_scores) >= 5 else (np.mean(hist_scores) if hist_scores else 0.0)
        sentiment_momentum   = today_score - avg_5d
        sentiment_volatility = np.std(hist_scores) if hist_scores else 0.0

        # News volume ratio
        days_with_signal = sum(1 for s in hist_scores if s != 0.0)
        avg_signal_rate  = max(days_with_signal / max(len(hist_scores), 1), 0.01)
        today_has_signal = 1 if today_score != 0.0 else 0
        sentiment_volume = today_has_signal / avg_signal_rate

        records.append({
            "symbol":               sym,
            "raw_sentiment":        today_score,
            "sentiment_momentum":   sentiment_momentum,
            "sentiment_volume":     sentiment_volume,
            "sentiment_volatility": sentiment_volatility,
        })

    df = pd.DataFrame(records)
    non_zero = (df["raw_sentiment"] != 0).sum()
    logger.info(f"[S2] Sentiment complete: {non_zero}/{len(df)} stocks with signals")
    return df


if __name__ == "__main__":
    from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols
    symbols = [s.replace(".NS", "") for s in get_nse_universe_symbols()]

    result = build_sentiment_features(date.today(), symbols[:30])
    non_zero = result[result["raw_sentiment"] != 0]
    print(f"\nSymbols with sentiment today: {len(non_zero)}")
    print(non_zero[["symbol", "raw_sentiment", "sentiment_momentum"]].to_string())