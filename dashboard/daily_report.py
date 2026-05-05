import json
import pandas as pd
from pathlib import Path
from datetime import date
from loguru import logger


def print_report():
    today = date.today().isoformat()

    # Load portfolio state
    state_file = Path("data/trades/portfolio_state.json")
    if not state_file.exists():
        print("No portfolio state found. Run the pipeline first.")
        return

    with open(state_file) as f:
        state = json.load(f)

    daily_log  = state.get("daily_log", [])
    trade_log  = state.get("trade_log", [])
    positions  = state.get("positions", {})

    latest = daily_log[-1] if daily_log else {}

    # Load regime
    regime_file = Path(f"data/processed/regime_{today}.json")
    regime = {}
    if regime_file.exists():
        with open(regime_file) as f:
            regime = json.load(f)

    # Closed trades
    closed = [t for t in trade_log if "exit_price" in t]
    wins   = [t for t in closed if t["pnl"] > 0]

    print(f"""
╔══════════════════════════════════════════════════════╗
║           🛰️  VEGAPUNK DAILY REPORT                   ║
║                   {today}                         ║
╠══════════════════════════════════════════════════════╣
║  MARKET                                              ║
║  Regime     : {regime.get('regime_label', 'N/A'):8}  (conf: {regime.get('confidence', 0):.2f})                 ║
║  Days in    : {regime.get('days_in_regime', 0):3} days                               ║
╠══════════════════════════════════════════════════════╣
║  PORTFOLIO                                           ║
║  Value      : ₹{latest.get('portfolio_value', 0):>12,.0f}                          ║
║  Return     : {latest.get('total_return_pct', 0):>+8.2f}%                              ║
║  Cash       : ₹{latest.get('cash', 0):>12,.0f}                          ║
║  Positions  : {latest.get('n_positions', 0):>3} open                               ║
║  CB Status  : {latest.get('circuit_breaker', 'N/A'):8}                               ║
╠══════════════════════════════════════════════════════╣
║  TODAY'S ACTIVITY                                    ║
║  Entries    : {latest.get('entries_today', 0):>3}                                    ║
║  Exits      : {latest.get('exits_today', 0):>3}                                    ║
╠══════════════════════════════════════════════════════╣
║  ALL-TIME STATS                                      ║
║  Total trades  : {len(closed):>3}                                 ║
║  Win rate      : {len(wins)/len(closed)*100 if closed else 0:>6.1f}%                             ║
║  Total P&L     : ₹{sum(t['pnl'] for t in closed):>+10,.0f}                         ║
╚══════════════════════════════════════════════════════╝""")

    # Open positions
    if positions:
        print(f"\n  OPEN POSITIONS ({len(positions)}):")
        print(f"  {'Symbol':12} {'Shares':>6} {'Entry':>8} {'Direction':>10}")
        print(f"  {'-'*42}")
        for sym, pos in positions.items():
            print(
                f"  {sym:12} {pos['shares']:>6} "
                f"₹{pos['entry_price']:>7.2f} {pos['direction']:>10}"
            )

    # Recent trades
    if closed:
        recent = sorted(closed, key=lambda x: x.get("date",""), reverse=True)[:5]
        print(f"\n  RECENT CLOSED TRADES:")
        print(f"  {'Symbol':12} {'P&L':>10} {'Days':>5} {'Reason':>15}")
        print(f"  {'-'*46}")
        for t in recent:
            print(
                f"  {t['symbol']:12} ₹{t['pnl']:>+9,.0f} "
                f"{t.get('days_held',0):>5}d "
                f"{t.get('exit_reason',''):>15}"
            )


if __name__ == "__main__":
    print_report()