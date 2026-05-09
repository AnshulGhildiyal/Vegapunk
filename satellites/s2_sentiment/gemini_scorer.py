"""
S2 — Gemini-Powered Sentiment Engine
Uses Google Gemini Flash to score news/announcements.
Gemini Flash: fast, cheap, excellent at financial text.
"""

import google.generativeai as genai
import json
import time
from loguru import logger

# Configure — get free key at https://aistudio.google.com/
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


SENTIMENT_PROMPT = """You are a financial analyst specializing in Indian stock markets (NSE/BSE).

Analyze the following corporate announcements and news for {symbol} and rate the overall sentiment impact on the stock price over the next 5 trading days.

Announcements/News:
{text}

Respond with ONLY a JSON object, no other text:
{{
  "score": <float between -1.0 and 1.0>,
  "reasoning": "<one sentence>",
  "key_signal": "<most important piece of news>"
}}

Scoring guide:
+0.8 to +1.0: Very bullish (strong earnings beat, major contract win, buyback announcement)
+0.4 to +0.8: Moderately bullish (good results, dividend, expansion plans)
0.0 to +0.4: Slightly positive (routine positive news)
-0.4 to 0.0: Slightly negative (minor concerns, delays)
-0.8 to -0.4: Moderately bearish (earnings miss, regulatory issues)
-1.0 to -0.8: Very bearish (fraud, major loss, delisting threat)
"""


def score_symbol_news(symbol: str, headlines: list[str]) -> dict:
    """
    Uses Gemini to score sentiment for a symbol given its recent headlines.
    Returns dict with score, reasoning, key_signal.
    """
    if not headlines:
        return {"score": 0.0, "reasoning": "No news", "key_signal": ""}

    text = "\n".join(f"- {h}" for h in headlines[:10])  # Max 10 headlines

    prompt = SENTIMENT_PROMPT.format(symbol=symbol, text=text)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,      # Low temp for consistent scoring
                max_output_tokens=200,
            )
        )

        raw = response.text.strip()
        # Clean up markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return {
            "score":      float(result.get("score", 0.0)),
            "reasoning":  result.get("reasoning", ""),
            "key_signal": result.get("key_signal", ""),
        }

    except json.JSONDecodeError:
        logger.warning(f"[S2] Gemini returned non-JSON for {symbol}")
        return {"score": 0.0, "reasoning": "Parse error", "key_signal": ""}
    except Exception as e:
        logger.warning(f"[S2] Gemini API error for {symbol}: {e}")
        return {"score": 0.0, "reasoning": "API error", "key_signal": ""}


def batch_score_symbols(
    symbol_headlines: dict[str, list[str]],
    delay_seconds: float = 0.5,
) -> dict[str, dict]:
    """
    Scores multiple symbols. Adds delay to respect rate limits.
    Gemini Flash free tier: 15 RPM = 4 seconds between calls to be safe.

    Args:
        symbol_headlines: {symbol: [headline1, headline2, ...]}
        delay_seconds: pause between API calls

    Returns:
        {symbol: {score, reasoning, key_signal}}
    """
    results = {}
    symbols_with_news = {k: v for k, v in symbol_headlines.items() if v}

    if not symbols_with_news:
        logger.info("[S2] No symbols with news to score")
        return {}

    logger.info(f"[S2] Scoring {len(symbols_with_news)} symbols with Gemini Flash")

    for i, (symbol, headlines) in enumerate(symbols_with_news.items(), 1):
        logger.debug(f"[S2] Scoring {symbol} ({i}/{len(symbols_with_news)})")
        results[symbol] = score_symbol_news(symbol, headlines)
        if i < len(symbols_with_news):
            time.sleep(delay_seconds)

    return results


if __name__ == "__main__":
    # Test with sample data
    test_data = {
        "RELIANCE": [
            "Reliance Industries reports 15% profit growth in Q4",
            "Reliance Jio adds 8 million subscribers in March",
            "Board approves Rs 9 per share dividend",
        ],
        "INFY": [
            "Infosys cuts revenue guidance for FY26",
            "Major client reduces contract value by 20%",
        ],
        "HDFCBANK": [
            "HDFC Bank Q4 results: NIM stable, loan growth 12%",
        ],
    }

    results = batch_score_symbols(test_data)
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Score  : {result['score']:+.2f}")
        print(f"  Signal : {result['key_signal']}")
        print(f"  Reason : {result['reasoning']}")