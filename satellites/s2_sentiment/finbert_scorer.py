"""
S2 — FinBERT Local Sentiment Scorer
Runs 100% locally. No API key. No cost. No rate limits.
ProsusAI/finbert is trained specifically on financial news.
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from loguru import logger
import time

# Load once at import time
logger.info("[S2] Loading FinBERT model (first run downloads ~400MB)...")
_tokenizer = None
_model = None

def _get_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        _model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
        logger.success("[S2] FinBERT loaded")
    return _tokenizer, _model


def score_text(text: str) -> float:
    """
    Scores a single text snippet.
    Returns float: +1.0 (very positive) to -1.0 (very negative)
    FinBERT outputs: [positive, negative, neutral]
    """
    tokenizer, model = _get_model()

    # Truncate to 512 tokens (BERT limit)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    # probs = [positive, negative, neutral]
    positive = float(probs[0])
    negative = float(probs[1])
    neutral  = float(probs[2])

    # Convert to -1 to +1 scale
    score = positive - negative
    return round(score, 4)


def score_symbol_news(symbol: str, headlines: list[str]) -> dict:
    """
    Scores all headlines for a symbol and aggregates.
    More recent headlines weighted higher.
    """
    if not headlines:
        return {"score": 0.0, "reasoning": "No news", "key_signal": ""}

    scores = []
    for i, headline in enumerate(headlines[:10]):
        try:
            s = score_text(headline)
            # Recency weight: first headline = most recent = higher weight
            weight = 1.0 / (1 + i * 0.2)
            scores.append((s, weight, headline))
        except Exception as e:
            logger.debug(f"[S2] Scoring failed for '{headline[:40]}': {e}")

    if not scores:
        return {"score": 0.0, "reasoning": "Scoring failed", "key_signal": ""}

    # Weighted average
    total_weight = sum(w for _, w, _ in scores)
    weighted_score = sum(s * w for s, w, _ in scores) / total_weight

    # Best signal = headline with highest absolute score
    best = max(scores, key=lambda x: abs(x[0]))

    return {
        "score":      round(weighted_score, 4),
        "reasoning":  f"Based on {len(scores)} headlines",
        "key_signal": best[2][:100],
    }


def batch_score_symbols(
    symbol_headlines: dict[str, list[str]],
    delay_seconds: float = 0.0,  # No delay needed — local model
) -> dict[str, dict]:
    """
    Scores multiple symbols using FinBERT locally.
    Much faster than API calls — no network latency.
    """
    results = {}
    symbols_with_news = {k: v for k, v in symbol_headlines.items() if v}

    if not symbols_with_news:
        return {}

    # Load model once before loop
    _get_model()
    logger.info(f"[S2] Scoring {len(symbols_with_news)} symbols with FinBERT")

    for i, (symbol, headlines) in enumerate(symbols_with_news.items(), 1):
        results[symbol] = score_symbol_news(symbol, headlines)
        logger.debug(
            f"[S2] {symbol}: {results[symbol]['score']:+.3f} "
            f"({i}/{len(symbols_with_news)})"
        )

    return results


if __name__ == "__main__":
    test_data = {
        "RELIANCE": [
            "Reliance Industries reports 15% profit growth in Q4",
            "Board approves Rs 9 per share dividend",
            "Reliance Jio adds 8 million subscribers",
        ],
        "INFY": [
            "Infosys cuts revenue guidance for FY26",
            "Major client reduces contract value by 20%",
        ],
        "HDFCBANK": [
            "HDFC Bank Q4 results: NIM stable, loan growth 12%",
        ],
        "BIOCON": [
            "Biocon shareholders meeting scheduled",
        ],
    }

    results = batch_score_symbols(test_data)
    print("\nFinBERT Sentiment Results:")
    print("=" * 45)
    for symbol, result in results.items():
        bar = "+" * int(max(0, result['score']) * 10) or "-" * int(max(0, -result['score']) * 10)
        print(f"{symbol:12} {result['score']:+.3f} {bar}")
        print(f"             {result['key_signal'][:60]}")