from dataclasses import dataclass
import yaml

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

TC = CONFIG["s6"]["transaction_costs"]


@dataclass
class TradeExecution:
    symbol:       str
    direction:    str 
    shares:       int
    fill_price:   float
    gross_value:  float 
    slippage:     float
    brokerage:    float
    stt:          float
    exchange_fee: float
    sebi_fee:     float
    gst:          float
    stamp_duty:   float
    total_cost:   float
    net_value:    float   


def compute_costs(
    direction:  str,
    shares:     int,
    price:      float,
    is_entry:   bool = True,
) -> TradeExecution:
    gross_value = shares * price

    # Slippage (market impact)
    slippage_pct = CONFIG["s6"]["slippage_pct"]
    if direction == "LONG":
        slippage = gross_value * slippage_pct      # Pay more when buying
    else:
        slippage = -gross_value * slippage_pct     # Receive less when selling

    # Brokerage (flat ₹20 per trade — Zerodha model)
    brokerage = TC["brokerage_per_trade"]

    # STT — only on sell side (0.1% of sell value)
    stt = gross_value * TC["stt_sell_pct"] if not is_entry else 0.0

    # Exchange transaction charges (both sides)
    exchange_fee = gross_value * TC["exchange_charges_pct"]

    # SEBI charges (both sides)
    sebi_fee = gross_value * TC["sebi_charges_pct"]

    # GST on brokerage (18%)
    gst = brokerage * TC["gst_on_brokerage_pct"]

    # Stamp duty (only on buy side)
    stamp_duty = gross_value * TC["stamp_duty_buy_pct"] if is_entry else 0.0

    total_cost = (
        abs(slippage) + brokerage + stt +
        exchange_fee + sebi_fee + gst + stamp_duty
    )

    net_value = gross_value + total_cost

    return TradeExecution(
        symbol      = "",
        direction   = direction,
        shares      = shares,
        fill_price  = price,
        gross_value = gross_value,
        slippage    = slippage,
        brokerage   = brokerage,
        stt         = stt,
        exchange_fee= exchange_fee,
        sebi_fee    = sebi_fee,
        gst         = gst,
        stamp_duty  = stamp_duty,
        total_cost  = total_cost,
        net_value   = net_value,
    )


def round_trip_cost_pct(price: float, shares: int) -> float:
    entry = compute_costs("LONG",  shares, price, is_entry=True)
    exit_ = compute_costs("SHORT", shares, price, is_entry=False)
    total = entry.total_cost + exit_.total_cost
    return total / (shares * price)


if __name__ == "__main__":
    # Test: ₹1L trade in RELIANCE at ₹2800
    price  = 2800.0
    shares = 35  # ~₹98,000 position

    entry = compute_costs("LONG", shares, price, is_entry=True)
    exit_ = compute_costs("SHORT", shares, price, is_entry=False)

    print(f"\nTransaction Cost Analysis — ₹{shares*price:,.0f} trade")
    print(f"Slippage      : ₹{entry.slippage:,.2f}")
    print(f"Brokerage     : ₹{entry.brokerage:,.2f} × 2")
    print(f"STT (exit)    : ₹{exit_.stt:,.2f}")
    print(f"Exchange fees : ₹{entry.exchange_fee + exit_.exchange_fee:,.2f}")
    print(f"SEBI fees     : ₹{entry.sebi_fee + exit_.sebi_fee:,.2f}")
    print(f"GST           : ₹{entry.gst + exit_.gst:,.2f}")
    print(f"Stamp duty    : ₹{entry.stamp_duty:,.2f}")
    total = entry.total_cost + exit_.total_cost
    print(f"TOTAL R/T cost: ₹{total:,.2f} ({total/(shares*price)*100:.3f}%)")
    print(f"\nBreakeven return needed: {total/(shares*price)*100:.3f}%")