import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date, timedelta
from loguru import logger

PROC_DIR  = Path("data/processed")
RAW_DIR   = Path("data/raw/bhavcopy")


def run_backtest(
    start_date: date = date(2025, 1, 1),
    end_date:   date = date(2025, 12, 31),
    initial_capital: float = 500_000,
):
    """
    Simulates paper trading on historical data.
    Measures: total return, Sharpe, max drawdown, win rate.
    """
    from satellites.s4_forecaster.xgb_model import predict_today
    from satellites.s7_regime.regime_detector import detect_regime

    capital   = initial_capital
    cash      = capital
    positions = {}
    daily_values = []
    trade_log = []

    logger.info(f"[BACKTEST] {start_date} → {end_date}")

    current = start_date
    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        feat_path = PROC_DIR / f"features_{current.isoformat()}.parquet"
        if not feat_path.exists():
            current += timedelta(days=1)
            continue

        features_df = pd.read_parquet(feat_path)

        # Get regime
        regime_path = PROC_DIR / f"regime_{current.isoformat()}.json"
        if regime_path.exists():
            with open(regime_path) as f:
                regime_data = json.load(f)
            regime_label = regime_data.get("regime_label", "RANGING")
        else:
            regime_label = "RANGING"

        # Current prices
        price_path = RAW_DIR / f"bhavcopy_{current.isoformat()}.parquet"
        if not price_path.exists():
            current += timedelta(days=1)
            continue

        prices_df = pd.read_parquet(price_path)
        prices = dict(zip(prices_df["symbol"], prices_df["close"]))

        # Check exits
        to_close = []
        for sym, pos in positions.items():
            price = prices.get(sym)
            if price is None:
                continue
            days_held = (current - date.fromisoformat(pos["entry_date"])).days
            pnl_pct   = (price - pos["entry_price"]) / pos["entry_price"]

            exit_reason = None
            stop_pct   = -0.02 if regime_label == "CRISIS" else -0.04
            hold_days  = 5    if regime_label == "CRISIS" else 10
            target_pct = 0.04 if regime_label == "CRISIS" else 0.06

            if pnl_pct <= stop_pct:
                exit_reason = "stop_loss"
            elif days_held >= hold_days:
                exit_reason = "max_hold"
            elif pnl_pct >= target_pct:
                exit_reason = "target_hit"

            if exit_reason:
                pnl = (price - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * price
                trade_log.append({
                    "symbol":      sym,
                    "entry_date":  pos["entry_date"],
                    "exit_date":   current.isoformat(),
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(pnl_pct * 100, 2),
                    "exit_reason": exit_reason,
                    "days_held":   days_held,
                })
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Get signals and enter
        signals = predict_today(features_df, regime_label)
        if not signals.empty:
            portfolio_value = cash + sum(
                pos["shares"] * prices.get(sym, pos["entry_price"])
                for sym, pos in positions.items()
            )

            min_signals = 1 if regime_label == "CRISIS" else 10

            for _, sig in signals.head(min_signals).iterrows():
                if sig["direction"] != "LONG":
                    continue
                if sig["symbol"] in positions:
                    continue
                if len(positions) >= 20:
                    break

                price = prices.get(sig["symbol"])
                if price is None:
                    continue

                position_value = min(portfolio_value * 0.04, cash * 0.5)
                shares = int(position_value // price)
                if shares == 0:
                    continue

                cost = shares * price
                if cost > cash:
                    continue

                cash -= cost
                positions[sig["symbol"]] = {
                    "shares":      shares,
                    "entry_price": price,
                    "entry_date":  current.isoformat(),
                }

        # Mark portfolio
        port_value = cash + sum(
            pos["shares"] * prices.get(sym, pos["entry_price"])
            for sym, pos in positions.items()
        )
        daily_values.append({
            "date":  current.isoformat(),
            "value": port_value,
        })

        current += timedelta(days=1)

    # Results
    daily_df = pd.DataFrame(daily_values)
    if daily_df.empty:
        logger.error("[BACKTEST] No data — run backfill_features.py first")
        return

    daily_df["ret"] = daily_df["value"].pct_change()
    total_ret  = (daily_df["value"].iloc[-1] / initial_capital - 1) * 100
    sharpe     = daily_df["ret"].mean() / daily_df["ret"].std() * np.sqrt(252)
    max_dd     = ((daily_df["value"].cummax() - daily_df["value"]) / daily_df["value"].cummax()).max() * 100

    closed = [t for t in trade_log if "exit_date" in t]
    wins   = [t for t in closed if t["pnl"] > 0]

    print(f"""
{'='*50}
VEGAPUNK BACKTEST RESULTS
{start_date} → {end_date}
{'='*50}
Total Return  : {total_ret:>+8.2f}%
Sharpe Ratio  : {sharpe:>8.2f}
Max Drawdown  : {max_dd:>8.2f}%
{'='*50}
Total Trades  : {len(closed):>6}
Win Rate      : {len(wins)/len(closed)*100 if closed else 0:>7.1f}%
Avg P&L/trade : ₹{np.mean([t['pnl'] for t in closed]) if closed else 0:>+8,.0f}
{'='*50}
Start Capital : ₹{initial_capital:>10,.0f}
End Capital   : ₹{daily_df['value'].iloc[-1]:>10,.0f}
{'='*50}
""")

    if closed:
        print("Exit reason breakdown:")
        reasons = pd.Series([t["exit_reason"] for t in closed]).value_counts()
        print(reasons.to_string())


# In scripts/backtest.py, add at the bottom:
if __name__ == "__main__":
    print("=== IN-SAMPLE TEST (2025) ===")
    run_backtest(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )

    print("\n=== OUT-OF-SAMPLE TEST (2026 YTD) ===")
    run_backtest(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 4, 30),
    )