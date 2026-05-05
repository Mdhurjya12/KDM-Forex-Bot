# executor.py
# KDM Trading System — Paper + Real Trading Execution Layer
# Optimized for NQ Futures (Prop Firm: Lucid / Topstep / Apex)

import json
import os
import csv
import ccxt
from datetime import datetime

# =========================
# TRADING MODE
# =========================
REAL_TRADING = False      # ← change to True when ready for real
# =========================
# PROP FIRM CONFIG
# NQ Futures rules (adjust to match your Lucid account rules)
# =========================
PROP_RULES = {
    "NASDAQ": {
        "max_daily_loss":    500,
        "max_total_loss":    2000,
        "max_contracts":     1,
        "profit_target":     3000,
    },
    "BTC": {
        "max_daily_loss":    300,
        "max_total_loss":    1000,
        "max_contracts":     1,
        "profit_target":     1000,
    },
    "GOLD": {
        "max_daily_loss":    200,
        "max_total_loss":    800,
        "max_contracts":     1,
        "profit_target":     1000,
    },
}

POINT_VALUE = {
    "NASDAQ": 2.0,
    "BTC":    1.0,
    "GOLD":   1.0,
}

BROKER_CONFIG = {
    "NASDAQ": {
        "broker":  "paper",
        "apiKey":  "",
        "secret":  "",
        "symbol":  "NQ=F",
    },
    "BTC": {
        "broker":  "binance",
        "apiKey":  "",
        "secret":  "",
        "symbol":  "BTC/USDT",
    },
    "GOLD": {
        "broker":  "binance",
        "apiKey":  "",
        "secret":  "",
        "symbol":  "PAXG/USDT",
    },
}

TRADE_LOG_FILE = "trade_log.csv"
STATE_FILE     = "bot_state.json"

def init_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "asset", "mode",
                "time_open", "time_close",
                "entry", "exit", "tp", "sl",
                "contracts", "outcome",
                "pnl", "balance", "notes"
            ])

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "balance":        10000.0,
        "open_trade":     None,
        "daily_pnl":      {},
        "total_trades":   0,
        "total_wins":     0,
        "total_losses":   0,
        "equity_curve":   [],
        "prop_progress":  0.0,
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def check_prop_rules(asset, state):
    rules    = PROP_RULES.get(asset, PROP_RULES["NASDAQ"])
    today    = datetime.now().strftime("%Y-%m-%d")
    day_loss = state["daily_pnl"].get(today, 0)

    if day_loss <= -rules["max_daily_loss"]:
        print(f"STOP [{asset}] DAILY LOSS LIMIT HIT — ${rules['max_daily_loss']}")
        print(f"   No more trades today. Prop firm rule enforced.")
        return False, "daily_loss_limit"

    start_bal = 10000.0
    drawdown  = start_bal - state["balance"]
    if drawdown >= rules["max_total_loss"]:
        print(f"STOP [{asset}] MAX DRAWDOWN HIT — ${rules['max_total_loss']}")
        return False, "max_drawdown"

    profit = state["balance"] - start_bal
    if profit >= rules["profit_target"]:
        print(f"TARGET [{asset}] PROFIT TARGET REACHED — ${rules['profit_target']}")
        print(f"   Consider stopping and submitting for payout!")

    return True, "ok"

def place_real_order(asset, signal, contracts):
    config = BROKER_CONFIG.get(asset)
    if not config or config["broker"] == "paper":
        return None

    try:
        if config["broker"] == "binance":
            exchange = ccxt.binance({
                "apiKey": config["apiKey"],
                "secret": config["secret"],
            })
            symbol = config["symbol"]
            if signal == "BUY":
                order = exchange.create_market_buy_order(symbol, contracts)
            else:
                order = exchange.create_market_sell_order(symbol, contracts)
            print(f"REAL ORDER PLACED: {order['id']} | {signal} {contracts}x {symbol}")
            return order

        print(f"Broker '{config['broker']}' not yet integrated.")
        print(f"Place manually: {signal} {contracts} contract(s) of NQ")
        return None

    except Exception as e:
        print(f"Real order failed: {e}")
        return None

def open_trade(asset, signal, price, tp_pct, sl_pct, prob):
    state = load_state()

    if state["open_trade"] is not None:
        print(f"[{asset}] Trade already open — skipping new signal")
        return False, "trade_already_open"

    allowed, reason = check_prop_rules(asset, state)
    if not allowed:
        return False, reason

    rules     = PROP_RULES.get(asset, PROP_RULES["NASDAQ"])
    contracts = rules["max_contracts"]

    if signal == "BUY":
        tp = price * (1 + tp_pct)
        sl = price * (1 - sl_pct)
    else:
        tp = price * (1 - tp_pct)
        sl = price * (1 + sl_pct)

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
        "contracts":  contracts,
        "time_open":  datetime.now().isoformat(),
        "mode":       "REAL" if REAL_TRADING else "PAPER",
    }

    save_state(state)

    mode_tag = "REAL" if REAL_TRADING else "PAPER"
    print(f"\n[{mode_tag}] [{asset}] {signal} OPENED")
    print(f"   Entry : {price:.2f}")
    print(f"   TP    : {tp:.2f}  (+{tp_pct*100:.2f}%)")
    print(f"   SL    : {sl:.2f}  (-{sl_pct*100:.2f}%)")
    print(f"   AI    : {prob:.0%} confidence")
    print(f"   Size  : {contracts} contract(s)\n")

    if REAL_TRADING:
        place_real_order(asset, signal, contracts)

    return True, "trade_opened"

def check_trade(current_high, current_low, current_price, asset):
    state = load_state()
    trade = state.get("open_trade")

    if trade is None:
        return None

    if trade["signal"] == "BUY":
        hit_tp = current_high >= trade["tp"]
        hit_sl = current_low  <= trade["sl"]
    else:
        hit_tp = current_low  <= trade["tp"]
        hit_sl = current_high >= trade["sl"]

    if hit_tp:
        return close_trade(trade["tp"], "WIN",  "TP hit", state)
    if hit_sl:
        return close_trade(trade["sl"], "LOSS", "SL hit", state)

    pv = POINT_VALUE.get(asset, 1.0)
    if trade["signal"] == "BUY":
        float_pnl = (current_price - trade["entry"]) * trade["contracts"] * pv
    else:
        float_pnl = (trade["entry"] - current_price) * trade["contracts"] * pv

    direction = "UP" if float_pnl >= 0 else "DOWN"
    print(f"[{asset}] Trade open {direction} | "
          f"Entry: {trade['entry']:.2f} | "
          f"Now: {current_price:.2f} | "
          f"Float: ${float_pnl:+.2f} | "
          f"TP: {trade['tp']:.2f} | "
          f"SL: {trade['sl']:.2f}")

    return None

def close_trade(exit_price, outcome, notes, state=None):
    if state is None:
        state = load_state()

    trade = state.get("open_trade")
    if trade is None:
        return None

    asset     = trade["asset"]
    pv        = POINT_VALUE.get(asset, 1.0)
    contracts = trade.get("contracts", 1)

    if trade["signal"] == "BUY":
        pnl = (exit_price - trade["entry"]) * contracts * pv
    else:
        pnl = (trade["entry"] - exit_price) * contracts * pv

    state["balance"]      += pnl
    state["total_trades"] += 1
    state["prop_progress"] = max(0, state["balance"] - 10000.0)

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

    init_trade_log()
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trade["id"],
            asset,
            trade.get("mode", "PAPER"),
            trade["time_open"],
            datetime.now().isoformat(),
            round(trade["entry"], 4),
            round(exit_price, 4),
            round(trade["tp"], 4),
            round(trade["sl"], 4),
            contracts,
            outcome,
            round(pnl, 4),
            round(state["balance"], 4),
            notes
        ])

    state["open_trade"] = None
    save_state(state)

    rules    = PROP_RULES.get(asset, PROP_RULES["NASDAQ"])
    progress = (state["prop_progress"] / rules["profit_target"]) * 100
    emoji    = "WIN" if outcome == "WIN" else "LOSS"

    print(f"\n[{emoji}] [{asset}] {outcome}")
    print(f"   Exit    : {exit_price:.2f}")
    print(f"   PnL     : ${pnl:+.2f}")
    print(f"   Balance : ${state['balance']:,.2f}")
    print(f"   Progress: {progress:.1f}% toward prop target\n")

    return {
        "outcome":  outcome,
        "pnl":      pnl,
        "balance":  state["balance"],
        "progress": progress
    }

def get_stats():
    state     = load_state()
    total     = state["total_trades"]
    wins      = state["total_wins"]
    today     = datetime.now().strftime("%Y-%m-%d")
    day_pnl   = state["daily_pnl"].get(today, 0)
    win_rate  = (wins / total * 100) if total > 0 else 0
    rules     = PROP_RULES.get("NASDAQ", PROP_RULES["NASDAQ"])
    progress  = (state.get("prop_progress", 0) / rules["profit_target"]) * 100

    return {
        "balance":         round(state["balance"], 2),
        "total_trades":    total,
        "wins":            wins,
        "losses":          state["total_losses"],
        "win_rate":        round(win_rate, 1),
        "daily_pnl":       round(day_pnl, 2),
        "open_trade":      state.get("open_trade"),
        "equity_curve":    state.get("equity_curve", []),
        "daily_breakdown": state.get("daily_pnl", {}),
        "prop_progress":   round(progress, 1),
        "mode":            "REAL" if REAL_TRADING else "PAPER",
    }
