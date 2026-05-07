from operator import pos
from unittest import signals

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date, datetime
from loguru import logger
import yaml

from satellites.s6_executor.transaction_costs import compute_costs
from satellites.s6_executor.position_sizer import compute_position_size

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S6_CFG   = CONFIG["s6"]
CB_CFG   = S6_CFG["circuit_breakers"]
TRADE_DIR = Path(CONFIG["paths"]["data_trades"])


class PaperTrader:
    """
    Simulates paper trading with full cost model and risk management.
    State persists across days via JSON.
    """

    STATE_FILE = Path("data/trades/portfolio_state.json")

    def __init__(self, initial_capital: float = None):
        self.initial_capital = (
            initial_capital or
            CONFIG["project"]["paper_trading_capital"]
        )
        self.state = self._load_state()

    # State Persistence 
    def _default_state(self) -> dict:
        return {
            "cash":            self.initial_capital,
            "initial_capital": self.initial_capital,
            "positions":       {},   # symbol → {shares, entry_price, entry_date, direction}
            "trade_log":       [],
            "daily_log":       [],
            "peak_value":      self.initial_capital,
            "circuit_breaker": "CLEAR",
        }

    def _load_state(self) -> dict:
        TRADE_DIR.mkdir(parents=True, exist_ok=True)
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                state = json.load(f)
            logger.info(
                f"[S6] Portfolio loaded: "
                f"cash=₹{state['cash']:,.0f}, "
                f"positions={len(state['positions'])}"
            )
            return state
        logger.info("[S6] New portfolio initialized")
        return self._default_state()

    def _save_state(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    # Portfolio Valuation
    def portfolio_value(self, current_prices: dict) -> float:
        """Total portfolio value: cash + mark-to-market positions."""
        position_value = sum(
            pos["shares"] * current_prices.get(sym, pos["entry_price"])
            for sym, pos in self.state["positions"].items()
        )
        return self.state["cash"] + position_value

    def daily_pnl(self, current_prices: dict) -> float:
        total_value = self.portfolio_value(current_prices)
        yesterday   = self.state["daily_log"][-1]["portfolio_value"] if self.state["daily_log"] else self.initial_capital
        return total_value - yesterday

    # Circuit Breakers 
    def check_circuit_breakers(self, current_prices: dict, run_date: str) -> str:
        """
        Checks all circuit breaker levels.
        Returns: 'CLEAR', 'LEVEL1', 'LEVEL2', 'LEVEL3'
        """
        total_value = self.portfolio_value(current_prices)
        daily_dd    = self.daily_pnl(current_prices) / self.initial_capital

        # Update peak
        if total_value > self.state["peak_value"]:
            self.state["peak_value"] = total_value

        weekly_dd = (self.state["peak_value"] - total_value) / self.state["peak_value"]

        if weekly_dd >= CB_CFG["level3_weekly_drawdown_pct"]:
            logger.warning(f"[S6] ⛔ CIRCUIT BREAKER LEVEL 3 — weekly DD {weekly_dd:.1%}")
            return "LEVEL3"
        elif daily_dd <= -CB_CFG["level2_daily_drawdown_pct"]:
            logger.warning(f"[S6] 🔴 CIRCUIT BREAKER LEVEL 2 — daily DD {daily_dd:.1%}")
            return "LEVEL2"
        elif daily_dd <= -CB_CFG["level1_daily_drawdown_pct"]:
            logger.warning(f"[S6] 🟡 CIRCUIT BREAKER LEVEL 1 — daily DD {daily_dd:.1%}")
            return "LEVEL1"
        return "CLEAR"

    # Exit Logic 

    def check_exits(self, current_prices: dict, run_date: str) -> list:
        """
        Checks all open positions for exit conditions.
        Returns list of exit trade records.
        """
        exits = []
        to_close = []

        for symbol, pos in self.state["positions"].items():
            price = current_prices.get(symbol)
            if price is None:
                continue

            entry_price = pos["entry_price"]
            entry_date  = pos["entry_date"]
            shares      = pos["shares"]

            pnl_pct = (price - entry_price) / entry_price
            if pos["direction"] == "SHORT":
                pnl_pct = -pnl_pct
                
            # Days held
            days_held = (
                pd.Timestamp(run_date) - pd.Timestamp(entry_date)
            ).days

            exit_reason = None

            # Stop loss
            if pnl_pct <= -S6_CFG["stop_loss_pct"]:
                exit_reason = "stop_loss"

            # Max holding period
            elif days_held >= S6_CFG["max_hold_days"]:
                exit_reason = "max_hold_period"

            # Trailing stop
            elif (pnl_pct >= S6_CFG["trailing_stop_trigger_pct"] and pnl_pct <= S6_CFG["trailing_stop_trigger_pct"] - S6_CFG["trailing_stop_retrace_pct"]):
                exit_reason = "trailing_stop"

            if exit_reason:
                if pos["direction"] == "LONG":
                    exit_exec = compute_costs("SHORT", shares, price, is_entry=False)
                    proceeds  = shares * price - exit_exec.total_cost
                    pnl       = proceeds - (shares * entry_price)
                else:  # SHORT
                    # Profit = entry price - exit price (we sold high, buy back low)
                    exit_exec = compute_costs("LONG", shares, price, is_entry=True)
                    proceeds  = shares * entry_price  # We already received this
                    buyback   = shares * price + exit_exec.total_cost
                    pnl       = proceeds - buyback

                trade_record = {
                    "date":         run_date,
                    "symbol":       symbol,
                    "direction":    pos["direction"],
                    "entry_price":  entry_price,
                    "exit_price":   price,
                    "shares":       shares,
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pnl_pct * 100, 3),
                    "exit_reason":  exit_reason,
                    "days_held":    days_held,
                    "costs":        round(exit_exec.total_cost, 2),
                }

                self.state["cash"] += proceeds
                self.state["trade_log"].append(trade_record)
                to_close.append(symbol)
                exits.append(trade_record)

                logger.info(
                    f"[S6] EXIT {symbol}: {exit_reason} | "
                    f"P&L ₹{pnl:+,.0f} ({pnl_pct:+.1%}) | "
                    f"held {days_held}d"
                )

        for sym in to_close:
            del self.state["positions"][sym]

        return exits

    # Entry Logic 

    def enter_positions(
        self,
        signals:        pd.DataFrame,
        current_prices: dict,
        regime:         str,
        run_date:       str,
    ) -> list:

        entries = []
        total_value = self.portfolio_value(current_prices)
        current_exp = 1 - (self.state["cash"] / total_value)

        if current_exp >= S6_CFG["max_total_exposure_pct"]:
            logger.info(
                f"[S6] Max exposure reached "
                f"({current_exp:.1%}) — no new entries"
            )
            return entries

        cb = self.state.get("circuit_breaker", "CLEAR")
        if cb in ["LEVEL2", "LEVEL3"]:
            logger.warning(f"[S6] Circuit breaker {cb} — no entries")
            return entries

        # In CRISIS regime — no new SHORT positions (too risky)
        allow_short = (regime != "CRISIS")

        for _, signal in signals.iterrows():
            symbol    = signal["symbol"]
            direction = signal["direction"]
            win_prob  = signal["xgb_pred"]
            price     = current_prices.get(symbol)

            if price is None or price <= 0:
                continue

            if symbol in self.state["positions"]:
                continue

        # Skip shorts in crisis or if not allowed
            if direction == "SHORT" and not allow_short:
                continue

            sizing = compute_position_size(
                capital        = total_value,
                win_prob       = win_prob,
                price          = price,
                regime         = regime,
            )

            if sizing["reason"] != "ok" or sizing["shares"] == 0:
                continue

            shares     = sizing["shares"]
            entry_exec = compute_costs(direction, shares, price, is_entry=True)
            cash_needed = shares * price + entry_exec.total_cost

            if cash_needed > self.state["cash"]:
                logger.warning(f"[S6] Insufficient cash for {symbol}")
                continue

            self.state["cash"] -= cash_needed
            self.state["positions"][symbol] = {
                "shares":      shares,
                "entry_price": price,
                "entry_date":  run_date,
                "direction":   direction,
                "win_prob":    round(win_prob, 4),
            }

            trade_record = {
                "date":           run_date,
                "symbol":         symbol,
                "direction":      direction,
                "entry_price":    price,
                "shares":         shares,
                "position_value": round(shares * price, 2),
                "pct_of_capital": sizing["pct_of_capital"],
                "costs":          round(entry_exec.total_cost, 2),
                "win_prob":       round(win_prob, 4),
                "regime":         regime,
            }

            self.state["trade_log"].append(trade_record)
            entries.append(trade_record)

            logger.info(
                f"[S6] ENTER {direction} {symbol}: "
                f"{shares} shares @ ₹{price:.2f} | "
                f"value=₹{shares*price:,.0f} | "
                f"prob={win_prob:.3f}"
            )

        return entries

    # Daily Run 

    def run_day(
        self,
        signals:        pd.DataFrame,
        current_prices: dict,
        regime:         str,
        run_date:       str,
    ) -> dict:
        """
        Main daily execution:
        1. Check circuit breakers
        2. Exit positions hitting stop/target/max_hold
        3. Enter new positions from signals
        4. Log daily state
        """
        total_value = self.portfolio_value(current_prices)
        cb_level = self.check_circuit_breakers(current_prices, run_date)
        self.state["circuit_breaker"] = cb_level

        exits   = self.check_exits(current_prices, run_date)
        entries = []

        if cb_level not in ["LEVEL2", "LEVEL3"] and signals is not None and not signals.empty:
            entries = self.enter_positions(signals, current_prices, regime, run_date)

        # Refresh valuation after trades
        total_value   = self.portfolio_value(current_prices)
        total_return  = (total_value - self.initial_capital) / self.initial_capital

        daily_record = {
            "date":            run_date,
            "portfolio_value": round(total_value, 2),
            "cash":            round(self.state["cash"], 2),
            "n_positions":     len(self.state["positions"]),
            "entries_today":   len(entries),
            "exits_today":     len(exits),
            "total_return_pct": round(total_return * 100, 3),
            "circuit_breaker": cb_level,
            "regime":          regime,
        }
        self.state["daily_log"].append(daily_record)
        self._save_state()

        logger.info(
            f"[S6] Daily summary: "
            f"portfolio=₹{total_value:,.0f} "
            f"({total_return:+.2%}) | "
            f"positions={len(self.state['positions'])} | "
            f"CB={cb_level}"
        )

        return {
            "entries":       len(entries),
            "exits":         len(exits),
            "portfolio_value": total_value,
            "total_return":  total_return,
            "circuit_breaker": cb_level,
        }


if __name__ == "__main__":
    trader = PaperTrader()
    print(f"\nPortfolio State:")
    print(f"  Cash: ₹{trader.state['cash']:,.0f}")
    print(f"  Positions: {len(trader.state['positions'])}")
    print(f"  Circuit breaker: {trader.state['circuit_breaker']}")