"""
S2 — NSE Announcement Scraper
Fetches company announcements from NSE.
These are high-quality, structured signals:
earnings results, board meetings, dividends, buybacks.
Free, legal, no rate limiting issues.
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
import time

RAW_DIR = Path("data/raw/sentiment/nse_announcements")
RAW_DIR.mkdir(parents=True, exist_ok=True)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_announcements(fetch_date: date) -> list[dict]:
    """
    Fetches NSE corporate announcements for a given date.
    Returns list of announcement dicts with symbol, headline, category.
    """
    local = RAW_DIR / f"announcements_{fetch_date.isoformat()}.json"

    if local.exists():
        with open(local) as f:
            data = json.load(f)
        logger.info(f"[S2] Loaded cached announcements: {len(data)} for {fetch_date}")
        return data

    date_str = fetch_date.strftime("%d-%m-%Y")
    url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={date_str}&to_date={date_str}"

    session = requests.Session()
    try:
        # Hit homepage first for cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(1)

        response = session.get(url, headers=NSE_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

    except Exception as e:
        logger.warning(f"[S2] NSE announcements fetch failed for {fetch_date}: {e}")
        return []

    # Normalize
    announcements = []
    for item in (data if isinstance(data, list) else data.get("data", [])):
        announcements.append({
            "symbol":    item.get("symbol", ""),
            "headline":  item.get("desc", item.get("subject", "")),
            "category":  item.get("an_type", ""),
            "date":      fetch_date.isoformat(),
            "exchange":  "NSE",
        })

    # Cache
    with open(local, "w") as f:
        json.dump(announcements, f, indent=2)

    logger.info(f"[S2] Fetched {len(announcements)} NSE announcements for {fetch_date}")
    return announcements


def categorize_announcement(headline: str, category: str) -> float:
    """
    Rule-based sentiment scoring for NSE announcements.
    No ML needed — announcements are structured and category tells you a lot.
    
    Returns score: +1.0 (very positive) to -1.0 (very negative)
    """
    headline_lower = headline.lower()
    category_lower = category.lower()

    # Strong positive signals
    if any(kw in headline_lower for kw in [
        "dividend", "buyback", "bonus", "split",
        "profit", "revenue growth", "strong results",
        "order win", "contract awarded", "capacity expansion",
        "acquisition completed", "merger approved",
    ]):
        return 0.7

    # Moderate positive
    if any(kw in headline_lower for kw in [
        "results", "financial results", "quarterly results",
        "board meeting", "agm", "rights issue",
    ]):
        return 0.3

    # Strong negative signals
    if any(kw in headline_lower for kw in [
        "loss", "default", "insolvency", "fraud",
        "sebi notice", "regulatory action", "shutdown",
        "delisting", "promoter pledge", "npa",
    ]):
        return -0.8

    # Moderate negative
    if any(kw in headline_lower for kw in [
        "penalty", "fine", "litigation", "legal notice",
        "rating downgrade", "revision",
    ]):
        return -0.4

    # Neutral / unknown
    return 0.0


def build_daily_sentiment(fetch_date: date, universe_symbols: list[str]) -> pd.DataFrame:
    """
    Builds per-symbol sentiment scores from NSE announcements.
    Returns DataFrame with symbol, sentiment_score, announcement_count.
    """
    announcements = fetch_nse_announcements(fetch_date)

    # Filter to universe symbols
    ann_df = pd.DataFrame(announcements)
    if ann_df.empty:
        # Return zeros for all universe symbols
        return pd.DataFrame({
            "symbol": universe_symbols,
            "raw_sentiment": 0.0,
            "announcement_count": 0,
        })

    ann_df = ann_df[ann_df["symbol"].isin(universe_symbols)]
    if ann_df.empty:
        return pd.DataFrame({
            "symbol": universe_symbols,
            "raw_sentiment": 0.0,
            "announcement_count": 0,
        })

    ann_df["score"] = ann_df.apply(
        lambda r: categorize_announcement(r["headline"], r["category"]), axis=1
    )

    # Aggregate per symbol
    agg = (ann_df.groupby("symbol").agg(raw_sentiment=("score", "mean"), announcement_count=("score", "count"),).reset_index())

    # Add missing symbols with zero sentiment
    all_symbols = pd.DataFrame({"symbol": universe_symbols})
    result = all_symbols.merge(agg, on="symbol", how="left")
    result["raw_sentiment"] = result["raw_sentiment"].fillna(0.0)
    result["announcement_count"] = result["announcement_count"].fillna(0).astype(int)

    logger.info(
        f"[S2] Sentiment built: {fetch_date} | "
        f"{(result['raw_sentiment'] != 0).sum()} stocks with signals"
    )
    return result


if __name__ == "__main__":
    today = date.today()
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK","SBIN", "AXISBANK", "HINDUNILVR", "ITC", "KOTAKBANK"]

    result = build_daily_sentiment(today, symbols)
    print(result[result["raw_sentiment"] != 0].to_string())
    print(f"\nTotal signals today: {(result['raw_sentiment'] != 0).sum()}")