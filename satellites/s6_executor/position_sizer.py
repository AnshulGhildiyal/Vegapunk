import numpy as np
import yaml
from loguru import logger

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S6_CFG = CONFIG["s6"]


def kelly_fraction(
    win_prob:      float,
    win_loss_ratio: float = 1.0,
    regime:        str = "RANGING",
) -> float:
    q = 1 - win_prob
    b = win_loss_ratio

    # Full Kelly
    f_star = (win_prob * b - q) / b

    # Fractional Kelly based on regime
    if regime == "CRISIS":
        fraction = S6_CFG["kelly_fraction_crisis"]   # 0.25
    else:
        fraction = S6_CFG["kelly_fraction"]          # 0.5

    f_half = f_star * fraction

    return max(0.0, f_half)


def compute_position_size(
    capital:       float,
    win_prob:      float,
    price:         float,
    regime:        str = "RANGING",
    win_loss_ratio: float = 1.0,
) -> dict:
    max_position_pct = S6_CFG["max_single_position_pct"]  # 5%
    max_exposure_pct = S6_CFG["max_total_exposure_pct"]    # 80%

    # Kelly-based position value
    f = kelly_fraction(win_prob, win_loss_ratio, regime)
    kelly_value = capital * f

    # Apply hard cap: max 5% of capital per position
    max_value = capital * max_position_pct
    position_value = min(kelly_value, max_value)

    # Must be at least 1 share
    if position_value < price:
        return {
            "shares":          0,
            "position_value":  0,
            "pct_of_capital":  0,
            "reason":          "position_too_small",
        }

    shares = int(position_value // price)
    actual_value = shares * price

    return {
        "shares":         shares,
        "position_value": round(actual_value, 2),
        "pct_of_capital": round(actual_value / capital * 100, 2),
        "kelly_f":        round(f, 4),
        "reason":         "ok",
    }


if __name__ == "__main__":
    capital = 500_000  # ₹5L paper trading capital

    print("Position Sizing Examples (₹5L capital)")
    print("=" * 55)

    test_cases = [
        ("RELIANCE", 2800, 0.63, "RANGING"),
        ("TCS",      3500, 0.58, "TRENDING"),
        ("INFY",     1800, 0.55, "CRISIS"),
        ("SBIN",      800, 0.52, "RANGING"),
    ]

    for symbol, price, win_prob, regime in test_cases:
        result = compute_position_size(capital, win_prob, price, regime)
        print(
            f"{symbol:12} | price=₹{price:>5} | prob={win_prob:.2f} | "
            f"regime={regime:8} | "
            f"shares={result['shares']:>4} | "
            f"value=₹{result['position_value']:>8,.0f} | "
            f"{result['pct_of_capital']:.1f}% of capital"
        )