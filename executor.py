# executor.py
# KDM Trading System — Paper Trading Execution Layer

import json
import os
import csv
from datetime import datetime

# =========================
# CONFIG
# =========================
TRADE_LOG_FILE  = "trade_log.csv"
STATE_FILE      = "bot_state.json"

PROP_RULES = {
    "BTC":    {"max_daily_loss": 500,   "max_contracts": 1},
    "GOLD":   {"max_daily_loss": 300,   "max_contracts": 1},
    "NASDAQ": {"max_daily_loss": 1000,  "max_contracts": 1},
}

POINT_VALUE = {
    "BTC":    1.0,
    "GOLD":   1.0,
    "NASDAQ": 2.0,   # MNQ = $2 per point
}

# =========================
# INIT TRADE LOG
# =========================
def init_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "asset", "time_open", "time_close",
                "entry", "exit", "tp", "sl",
                "outcome", "pnl", "balance", "notes"
            ])

# =========================
# STATE MANAGEMENT
# =========================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "balance":      10000.0,
        "open_trade":   None,
        "daily_pnl":    {},
        "total_trades": 0,
        "total_wins":   0,
        "total_losses": 0,
        "equity_curve": [],
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# =========================
# OPEN TRADE
# =========================
def open_trade(asset, signal, price, tp_pct, sl_pct, prob):
    state = load_state()

    # Block if trade already open
    if state["open_trade"] is not None:
        return False, "Trade already open"

    # Check daily loss limit
    today    = datetime.now().strftime("%Y-%m-%d")
    day_loss = state["daily_pnl"].get(today, 0)
    rules    = PROP_RULES.get(asset, PROP_RULES["NASDAQ"])

    if day_loss <= -rules["max_daily_loss"]:
        return False, f"Daily loss limit hit (${rules['max_daily_loss']})"

    tp = price * (1 + tp_pct) if signal == "BUY" else price * (1 - tp_pct)
    sl = price * (1 - sl_pct) if signal == "BUY" else price * (1 + sl_pct)

    state["open_trade"] = {
        "id":         state["total_trades"] + 1,
        "asset":      asset,
        "signal":     signal,
        "entry":      price,
        "tp":         tp,
        "sl":         sl,
        "tp_pct":     tp_pct,
        "sl_pct":     sl_pct,
        "prob":       prob,
        "time_open":  datetime.now().isoformat(),
        "contracts":  rules["max_contracts"],
    }

    save_state(state)
    print(f"📈 [{asset}] {signal} opened @ {price:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | AI: {prob:.0%}")
    return True, "Trade opened"

# =========================
# CHECK & CLOSE TRADE
# =========================
def check_trade(current_high, current_low, current_price, asset):
    state = load_state()
    trade = state.get("open_trade")

    if trade is None:
        return None

    hit_tp = current_high >= trade["tp"] if trade["signal"] == "BUY" else current_low <= trade["tp"]
    hit_sl = current_low  <= trade["sl"] if trade["signal"] == "BUY" else current_high >= trade["sl"]

    if hit_tp:
        return close_trade(trade["tp"], "WIN",  "TP hit", state)
    if hit_sl:
        return close_trade(trade["sl"], "LOSS", "SL hit", state)

    # Show floating PnL
    pv        = POINT_VALUE.get(asset, 1.0)
    float_pnl = (current_price - trade["entry"]) * trade["contracts"] * pv
    if trade["signal"] == "SELL":
        float_pnl = -float_pnl

    print(f"⏳ [{asset}] Trade open | Entry: {trade['entry']:.2f} | "
          f"Current: {current_price:.2f} | Float PnL: ${float_pnl:+.2f}")
    return None

# =========================
# CLOSE TRADE
# =========================
def close_trade(exit_price, outcome, notes, state=None):
    if state is None:
        state = load_state()

    trade = state.get("open_trade")
    if trade is None:
        return None

    asset = trade["asset"]
    pv    = POINT_VALUE.get(asset, 1.0)

    if trade["signal"] == "BUY":
        pnl = (exit_price - trade["entry"]) * trade["contracts"] * pv
    else:
        pnl = (trade["entry"] - exit_price) * trade["contracts"] * pv

    state["balance"]      += pnl
    state["total_trades"] += 1

    today = datetime.now().strftime("%Y-%m-%d")
    state["daily_pnl"][today] = state["daily_pnl"].get(today, 0) + pnl

    if outcome == "WIN":
        state["total_wins"]   += 1
    else:
        state["total_losses"] += 1

    state["equity_curve"].append({
        "time":    datetime.now().isoformat(),
        "balance": round(state["balance"], 2)
    })

    # Log to CSV
    init_trade_log()
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trade["id"], asset,
            trade["time_open"], datetime.now().isoformat(),
            round(trade["entry"], 4), round(exit_price, 4),
            round(trade["tp"], 4), round(trade["sl"], 4),
            outcome, round(pnl, 4),
            round(state["balance"], 4), notes
        ])

    state["open_trade"] = None
    save_state(state)

    emoji = "✅" if outcome == "WIN" else "❌"
    print(f"{emoji} [{asset}] {outcome} | Exit: {exit_price:.2f} | "
          f"PnL: ${pnl:+.2f} | Balance: ${state['balance']:,.2f}")

    return {"outcome": outcome, "pnl": pnl, "balance": state["balance"]}

# =========================
# GET STATS
# =========================
def get_stats():
    state = load_state()
    total = state["total_trades"]
    wins  = state["total_wins"]

    today     = datetime.now().strftime("%Y-%m-%d")
    day_pnl   = state["daily_pnl"].get(today, 0)
    win_rate  = (wins / total * 100) if total > 0 else 0

    return {
        "balance":      round(state["balance"], 2),
        "total_trades": total,
        "wins":         wins,
        "losses":       state["total_losses"],
        "win_rate":     round(win_rate, 1),
        "daily_pnl":    round(day_pnl, 2),
        "open_trade":   state.get("open_trade"),
        "equity_curve": state.get("equity_curve", []),
        "daily_breakdown": state.get("daily_pnl", {}),
    }