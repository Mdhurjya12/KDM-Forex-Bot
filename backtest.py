# backtest.py
# KDM Trading System — Historical Backtester

import ccxt
import pandas as pd
from strategy import add_ema, generate_signal, label_candles
import sys

ASSET_CONFIGS = {
    "BTC":    {"symbol": "BTC/USDT", "exchange": "binance",   "tp": 0.002, "sl": 0.002},
    "GOLD":   {"symbol": "PAXG/USDT","exchange": "binance",   "tp": 0.001, "sl": 0.001},
    "NASDAQ": {"symbol": "NQ=F",     "exchange": "yfinance",  "tp": 0.0005,"sl": 0.0005},
}

if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in ASSET_CONFIGS:
        ACTIVE_ASSET = arg
    else:
        print(f"❌  Unknown asset. Choose: {list(ASSET_CONFIGS.keys())}")
        sys.exit(1)
else:
    ACTIVE_ASSET = "BTC"

# =========================
# CONFIG
# =========================
SYMBOL         = "BTC/USDT"
TIMEFRAME      = "1m"
LIMIT          = 1000
INITIAL_BAL    = 1000.0

TP_PCT         = 0.006  
SL_PCT         = 0.003    
RISK_PER_TRADE = 0.01

# =========================
# FETCH HISTORICAL DATA
# =========================
def fetch_historical(symbol, timeframe, limit):
    exchange = ccxt.binance()
    ohlcv    = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# =========================
# RUN BACKTEST
# =========================
def run_backtest():
    print("Fetching historical data...")
    df = fetch_historical(SYMBOL, TIMEFRAME, LIMIT)
    print(f"   {len(df)} candles loaded")

    # Indicators computed once on full df
    df = add_ema(df)
    df["ema_distance"] = df["ema9"] - df["ema15"]
    df["volatility"]   = df["close"].rolling(10).std()

    # Smart labels computed once on full df
    df = label_candles(df, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead=20)

    # Clean
    df = df.dropna(subset=["ema9", "ema15", "ema_distance", "volatility"])
    df = df[df["label"] != -1].reset_index(drop=True)

    print(f"   {len(df)} candles after cleaning")

    # Count signals for debug
    signal_count = 0
    for i in range(len(df)):
        if generate_signal(df.iloc[:i+1]) == "BUY":
            signal_count += 1
    print(f"   {signal_count} BUY signals detected\n")

    # Simulate trades
    balance = INITIAL_BAL
    trades  = []
    wins    = 0
    losses  = 0

    for i in range(len(df)):
        signal = generate_signal(df.iloc[:i+1])

        if signal != "BUY":
            continue

        row   = df.iloc[i]
        entry = row["close"]
        size  = balance * RISK_PER_TRADE

        if row["label"] == 1:
            pnl     = size * TP_PCT
            wins   += 1
            outcome = "WIN"
        else:
            pnl     = -(size * SL_PCT)
            losses += 1
            outcome = "LOSS"

        balance += pnl

        trades.append({
            "time":    row["timestamp"],
            "entry":   round(entry, 2),
            "outcome": outcome,
            "pnl":     round(pnl, 4),
            "balance": round(balance, 4)
        })

    # Results
    total    = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    net_pnl  = balance - INITIAL_BAL

    print("=" * 42)
    print("  KDM Backtest Results")
    print("=" * 42)
    print(f"  Candles analysed : {len(df)}")
    print(f"  Total trades     : {total}")
    print(f"  Wins             : {wins}  ({win_rate:.1f}%)")
    print(f"  Losses           : {losses}")
    print(f"  Start balance    : ${INITIAL_BAL:.2f}")
    print(f"  End balance      : ${balance:.2f}")
    print(f"  Net PnL          : ${net_pnl:+.2f}")
    print("=" * 42)

    if total > 0:
        pd.DataFrame(trades).to_csv("backtest_results.csv", index=False)
        print("  Saved → backtest_results.csv\n")
        print("  First 5 trades:")
        print(pd.DataFrame(trades).head()[["time","entry","outcome","pnl","balance"]].to_string(index=False))
    else:
        print("  No trades found — check your strategy signals\n")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_backtest()