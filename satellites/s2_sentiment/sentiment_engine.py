"""
S2 — Full Sentiment Engine
Combines NSE announcements + Gemini scoring + rolling aggregation.
Produces 5 sentiment features for S3.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger

from satellites.s2_sentiment.nse_announcements import fetch_nse_announcements
from satellites.s2_sentiment.finbert_scorer import batch_score_symbols

SENTIMENT_DIR = Path("data/raw/sentiment/scores")
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


def get_rolling_headlines(
    symbol: str,
    end_date: date,
    lookback_days: int = 7,
) -> list[str]:
    """Gets all headlines for a symbol over the past N days."""
    headlines = []
    for i in range(lookback_days):
        d = end_date - timedelta(days=i)
        anns = fetch_nse_announcements(d)
        for ann in anns:
            if ann["symbol"] == symbol and ann["headline"]:
                headlines.append(f"[{d}] {ann['headline']}")
    return headlines


def build_sentiment_features(
    run_date: date,
    universe_symbols: list[str],
) -> pd.DataFrame:
    """
    Main S2 function. Builds full sentiment feature matrix.

    Features:
    - raw_sentiment: Gemini score for today's news
    - sentiment_momentum: today vs 5-day average
    - sentiment_volume: news count vs 30-day average
    - price_sentiment_divergence: computed in S3
    - sentiment_volatility: std of scores over 10 days
    """
    score_path = SENTIMENT_DIR / f"scores_{run_date.isoformat()}.json"

    # Use cached scores if available
    if score_path.exists():
        with open(score_path) as f:
            cached = json.load(f)
        logger.info(f"[S2] Loaded cached scores for {run_date}: {len(cached)} symbols")
    else:
        # Find which symbols have news today
        today_anns = fetch_nse_announcements(run_date)
        symbols_with_news = set(a["symbol"] for a in today_anns)

        # Build headline dict for symbols that have news
        symbol_headlines = {}
        for sym in universe_symbols:
            if sym in symbols_with_news:
                headlines = get_rolling_headlines(sym, run_date, lookback_days=7)
                if headlines:
                    symbol_headlines[sym] = headlines

        logger.info(
            f"[S2] {len(symbol_headlines)} of {len(universe_symbols)} "
            f"symbols have news for {run_date}"
        )

        # Score with Gemini (only symbols with news)
        if symbol_headlines:
            scores = batch_score_symbols(symbol_headlines, delay_seconds=1.0)
        else:
            scores = {}

        # Save scores
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=2)

        cached = scores

    # Build feature DataFrame
    records = []
    for sym in universe_symbols:
        today_score = cached.get(sym, {}).get("score", 0.0)

        # Load historical scores for momentum/volatility
        historical_scores = []
        for i in range(1, 11):  # Last 10 days
            hist_path = SENTIMENT_DIR / f"scores_{(run_date - timedelta(days=i)).isoformat()}.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                historical_scores.append(hist.get(sym, {}).get("score", 0.0))

        # Compute derived features
        if historical_scores:
            avg_5d = np.mean(historical_scores[:5]) if len(historical_scores) >= 5 else np.mean(historical_scores)
            sentiment_momentum = today_score - avg_5d
            sentiment_volatility = np.std(historical_scores)
        else:
            sentiment_momentum = 0.0
            sentiment_volatility = 0.0

        # News volume (count of days with news in last 30 days)
        news_days = sum(
            1 for i in range(30)
            if (SENTIMENT_DIR / f"scores_{(run_date - timedelta(days=i)).isoformat()}.json").exists()
            and (
                json.load(open(SENTIMENT_DIR / f"scores_{(run_date - timedelta(days=i)).isoformat()}.json"))
                .get(sym, {}).get("score", None) is not None
                and json.load(open(SENTIMENT_DIR / f"scores_{(run_date - timedelta(days=i)).isoformat()}.json"))
                .get(sym, {}).get("score", 0.0) != 0.0
            )
        )
        avg_news_days = max(news_days / 30, 0.01)
        today_has_news = 1 if today_score != 0.0 else 0
        sentiment_volume = today_has_news / avg_news_days

        records.append({
            "symbol":                    sym,
            "raw_sentiment":             today_score,
            "sentiment_momentum":        sentiment_momentum,
            "sentiment_volume":          sentiment_volume,
            "sentiment_volatility":      sentiment_volatility,
        })

    df = pd.DataFrame(records)
    logger.info(
        f"[S2] Features built: {(df['raw_sentiment'] != 0).sum()} "
        f"stocks with non-zero sentiment"
    )
    return df


if __name__ == "__main__":
    from satellites.s1_universe.bhavcopy_downloader import get_nse_universe_symbols
    symbols = [s.replace(".NS", "") for s in get_nse_universe_symbols()][:20]

    result = build_sentiment_features(date.today(), symbols)
    non_zero = result[result["raw_sentiment"] != 0]
    if not non_zero.empty:
        print("\nSymbols with sentiment today:")
        print(non_zero[["symbol", "raw_sentiment", "sentiment_momentum"]].to_string())
    else:
        print("No sentiment signals today (no NSE announcements for universe stocks)")