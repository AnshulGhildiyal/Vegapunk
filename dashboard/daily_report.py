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

    def get_current_prices() -> dict:
        today = date.today().isoformat()
        feature_path = Path(f"data/processed/features_{today}.parquet")

        if feature_path.exists():
            df = pd.read_parquet(feature_path)
            if "close" in df.columns:
                return dict(zip(df["symbol"], df["close"]))

        # Fallback: most recent universe
        universe_files = sorted(Path("data/universes").glob("universe_*.parquet"))
        if universe_files:
            u = pd.read_parquet(universe_files[-1])
            return dict(zip(u["symbol"], u["close"]))
        return {}

    current_prices = get_current_prices()

    # Closed trades
    closed = [t for t in trade_log if "exit_price" in t]
    wins   = [t for t in closed if t["pnl"] > 0]

    open_pnl = 0
    for sym, pos in positions.items():
        price = current_prices.get(sym, pos["entry_price"])
        if pos["direction"] == "LONG":
            open_pnl += (price - pos["entry_price"]) * pos["shares"]
        else:
            open_pnl += (pos["entry_price"] - price) * pos["shares"]

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
║  Open P&L   : ₹{open_pnl:>+12,.0f}                          ║
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

    if closed:
        stop_loss_trades = [t for t in closed if t.get("exit_reason") == "stop_loss"]
        max_hold_trades  = [t for t in closed if t.get("exit_reason") == "max_hold_period"]

        sl_wins = sum(1 for t in stop_loss_trades if t.get("pnl", 0) > 0)
        mh_wins = sum(1 for t in max_hold_trades  if t.get("pnl", 0) > 0)

        print(f"  Exit breakdown:")
        print(f"  Stop-loss : {len(stop_loss_trades):>3} trades | {sl_wins} wins ({sl_wins/max(len(stop_loss_trades),1)*100:.0f}%)")
        print(f"  Max-hold  : {len(max_hold_trades):>3} trades | {mh_wins} wins ({mh_wins/max(len(max_hold_trades),1)*100:.0f}%)")
    
    # Open positions
    
    if positions:
        print(f"\n  OPEN POSITIONS ({len(positions)}) — Mark-to-Market:")
        print(f"  {'Symbol':12} {'Dir':5} {'Shares':>6} {'Entry':>8} {'Now':>8} {'P&L':>10} {'Days':>5}")
        print(f"  {'-'*60}")

        today_ts = pd.Timestamp(today)
        for sym, pos in sorted(positions.items(),
                                key=lambda x: (
                                    current_prices.get(x[0], x[1]['entry_price'])
                                    - x[1]['entry_price']
                                ) * x[1]['shares'],
                                reverse=True):
            price = current_prices.get(sym, pos["entry_price"])
            if pos["direction"] == "LONG":
                pnl = (price - pos["entry_price"]) * pos["shares"]
            else:
                pnl = (pos["entry_price"] - price) * pos["shares"]

            days = (today_ts - pd.Timestamp(pos["entry_date"])).days

            print(
                f"  {sym:12} {pos['direction']:5} "
                f"{pos['shares']:>6} "
                f"₹{pos['entry_price']:>7.2f} "
                f"₹{price:>7.2f} "
                f"₹{pnl:>+9,.0f} "
                f"{days:>4}d"
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
    if len(daily_log) > 1:
        print("\n  EQUITY CURVE (last 20 days):")
        recent = daily_log[-20:]
        max_val = max(d.get("portfolio_value", 500000) for d in recent)
        min_val = min(d.get("portfolio_value", 500000) for d in recent)
        val_range = max_val - min_val if max_val != min_val else 1

        for entry in recent:
            val = entry.get("portfolio_value", 500000)
            ret = entry.get("total_return_pct", 0)
            bar_len = int((val - min_val) / val_range * 30)
            bar = "*" * bar_len
            sign = "+" if ret >= 0 else ""
            date_str = entry.get("date", "")[-5:]
            print(f"  {date_str}  {bar:<30} {sign}{ret:.2f}%")

if __name__ == "__main__":
    print_report()