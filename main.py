# main.py
# KDM Trading System — Multi-Asset Bot with Execution

import sys
import time
import ccxt
import pandas as pd
from datetime import datetime
from strategy import add_ema, generate_signal, label_candles
from ai_filter import ai_filter
from executor import open_trade, check_trade, get_stats, init_trade_log

# =========================
# ASSET CONFIG
# =========================
ASSETS = {
    "BTC": {
        "symbol":     "BTC/USDT",
        "exchange":   "binance",
        "timeframe":  "1m",
        "tp_pct":     0.001,
        "sl_pct":     0.001,
        "data_file":  "ai_data_btc.csv",
        "model_file": "kdm_model_btc.pkl",
        "scaler_file":"kdm_scaler_btc.pkl",
    },
    "GOLD": {
        "symbol":     "PAXG/USDT",
        "exchange":   "binance",
        "timeframe":  "1m",
        "tp_pct":     0.0008,
        "sl_pct":     0.0008,
        "data_file":  "ai_data_gold.csv",
        "model_file": "kdm_model_gold.pkl",
        "scaler_file":"kdm_scaler_gold.pkl",
    },
    "NASDAQ": {
        "symbol":     "NQ=F",
        "exchange":   "yfinance",
        "timeframe":  "1m",
        "tp_pct":     0.0003,
        "sl_pct":     0.0003,
        "data_file":  "ai_data_nasdaq.csv",
        "model_file": "kdm_model_nasdaq.pkl",
        "scaler_file":"kdm_scaler_nasdaq.pkl",
    },
}

# ── SWITCH ASSET FROM TERMINAL ────────────────────
# Usage:
#   python3 main.py          ← defaults to BTC
#   python3 main.py GOLD
#   python3 main.py NASDAQ

if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in ASSETS:
        ACTIVE_ASSET = arg
    else:
        print(f"❌  Unknown asset '{arg}'. Choose: {list(ASSETS.keys())}")
        sys.exit(1)
else:
    ACTIVE_ASSET = "BTC"

# Load active config
CONFIG       = ASSETS[ACTIVE_ASSET]
SYMBOL       = CONFIG["symbol"]
TIMEFRAME    = CONFIG["timeframe"]
TP_PCT       = CONFIG["tp_pct"]
SL_PCT       = CONFIG["sl_pct"]
DATA_FILE    = CONFIG["data_file"]
CANDLE_LIMIT = 100
LOOKAHEAD    = 40

# =========================
# EXCHANGE SETUP
# =========================
def get_exchange():
    if CONFIG["exchange"] == "binance":
        return ccxt.binance()
    return None

# =========================
# FETCH CANDLES
# =========================
def fetch_candles():
    if CONFIG["exchange"] == "yfinance":
        return fetch_yfinance()

    exchange = get_exchange()
    ohlcv    = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=CANDLE_LIMIT)

    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )
    return df


def fetch_yfinance():
    try:
        import yfinance as yf
    except ImportError:
        print("❌  yfinance not installed. Run: pip install yfinance")
        return None

    ticker = yf.Ticker(SYMBOL)
    df     = ticker.history(period="1d", interval="1m")

    if df.empty:
        print(f"❌  No data returned for {SYMBOL}")
        return None

    df = df.reset_index()
    df = df.rename(columns={
        "Datetime": "time",
        "Open":     "open",
        "High":     "high",
        "Low":      "low",
        "Close":    "close",
        "Volume":   "volume"
    })

    return df[["time", "open", "high", "low", "close", "volume"]].tail(CANDLE_LIMIT)


# =========================
# INIT
# =========================
init_trade_log()

try:
    pd.read_csv(DATA_FILE)
    print(f"📂 Existing data file found: {DATA_FILE}")
except FileNotFoundError:
    pd.DataFrame(
        columns=["ema9", "ema15", "ema_distance", "volatility", "label"]
    ).to_csv(DATA_FILE, index=False)
    print(f"📂 Created new data file: {DATA_FILE}")

print(f"\n{'='*50}")
print(f"  KDM Bot — {ACTIVE_ASSET} ({SYMBOL})")
print(f"  TP: {TP_PCT*100}%  SL: {SL_PCT*100}%  Lookahead: {LOOKAHEAD}")
print(f"{'='*50}\n")

# =========================
# MAIN LOOP
# =========================
while True:
    try:
        # ── FETCH ─────────────────────────────────────────
        df = fetch_candles()

        if df is None or df.empty:
            print("⚠️  No data received. Retrying...")
            time.sleep(30)
            continue

        # ── INDICATORS ────────────────────────────────────
        df = add_ema(df)
        df["ema_distance"] = df["ema9"] - df["ema15"]
        df["volatility"]   = df["close"].rolling(10).std()

        # ── SMART LABELING ────────────────────────────────
        df = label_candles(df, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead=LOOKAHEAD)

        df = df.dropna(subset=["ema9", "ema15", "ema_distance", "volatility"])
        df = df[df["label"] != -1].reset_index(drop=True)

        if df.empty:
            print("⚠️  Not enough candles yet. Waiting...")
            time.sleep(60)
            continue

        # ── SIGNAL ────────────────────────────────────────
        signal = generate_signal(df, asset=ACTIVE_ASSET)
        last   = df.iloc[-1]

        ema_dist = float(last["ema_distance"])
        vol      = float(last["volatility"])
        label    = int(last["label"])

        # ── CHECK OPEN TRADE ──────────────────────────────
        # Always check first if an existing trade hit TP or SL
        check_trade(
            current_high  = float(last["high"]),
            current_low   = float(last["low"]),
            current_price = float(last["close"]),
            asset         = ACTIVE_ASSET
        )

        # ── AI FILTER + EXECUTION ─────────────────────────
        if signal == "BUY":
            allow, prob = ai_filter(
                ema9         = float(last["ema9"]),
                ema15        = float(last["ema15"]),
                ema_distance = ema_dist,
                volatility   = vol,
                asset        = ACTIVE_ASSET
            )

            if allow:
                print(f"✅  TRADE APPROVED — {ACTIVE_ASSET} | confidence: {prob:.0%}")
                open_trade(
                    asset  = ACTIVE_ASSET,
                    signal = "BUY",
                    price  = float(last["close"]),
                    tp_pct = TP_PCT,
                    sl_pct = SL_PCT,
                    prob   = prob
                )
            else:
                print(f"🚫  TRADE BLOCKED  — {ACTIVE_ASSET} | confidence: {prob:.0%}")

        elif signal == "SELL":
            allow, prob = ai_filter(
                ema9         = float(last["ema9"]),
                ema15        = float(last["ema15"]),
                ema_distance = ema_dist,
                volatility   = vol,
                asset        = ACTIVE_ASSET
            )

            if allow:
                print(f"✅  SELL APPROVED  — {ACTIVE_ASSET} | confidence: {prob:.0%}")
                open_trade(
                    asset  = ACTIVE_ASSET,
                    signal = "SELL",
                    price  = float(last["close"]),
                    tp_pct = TP_PCT,
                    sl_pct = SL_PCT,
                    prob   = prob
                )
            else:
                print(f"🚫  SELL BLOCKED   — {ACTIVE_ASSET} | confidence: {prob:.0%}")

        # ── DISPLAY ───────────────────────────────────────
        stats = get_stats()
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{ACTIVE_ASSET:<6} | "
            f"Price: {float(last['close']):,.2f} | "
            f"EMA Dist: {ema_dist:+.4f} | "
            f"Signal: {signal:<8} | "
            f"Label: {'✅ WIN' if label == 1 else '❌ LOSS'} | "
            f"Balance: ${stats['balance']:,.2f} | "
            f"WR: {stats['win_rate']}%"
        )

        # ── SAVE TO AI DATA ───────────────────────────────
        new_row = pd.DataFrame([{
            "ema9":         round(float(last["ema9"]),  6),
            "ema15":        round(float(last["ema15"]), 6),
            "ema_distance": round(ema_dist,             6),
            "volatility":   round(vol,                  6),
            "label":        label
        }])

        new_row.to_csv(DATA_FILE, mode="a", header=False, index=False)

        time.sleep(60)

    except Exception as e:
        print(f"⚠️  Error: {e}")
        time.sleep(5)