# main.py
# KDM Trading System — Multi-Asset Bot with VWAP + Order Block
# Sound alerts included — no need to watch screen all day

import sys
import time
import subprocess
import ccxt
import pandas as pd
from datetime import datetime
from strategy import add_ema, generate_signal, label_candles, detect_order_blocks, detect_fractal_sweep, detect_cisd
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

# =========================
# SESSION FILTER
# Set to True  → only signals during market hours
# Set to False → signals 24/7 (good for testing)
# =========================
USE_SESSION_FILTER = False    # ← change to True for live prop trading

# =========================
# SWITCH ASSET FROM TERMINAL
# python3 main.py          ← BTC
# python3 main.py GOLD     ← Gold
# python3 main.py NASDAQ   ← Nasdaq
# =========================
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
# SOUND ALERT
# Plays Mac system sound when signal fires
# =========================
def send_alert(message, sound="Glass"):
    print(f"\n🔔  ALERT: {message}\n")
    try:
        subprocess.run(
            ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
            timeout=3
        )
    except Exception:
        pass   # silent fail if sound not available

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

print(f"\n{'='*55}")
print(f"  KDM Bot — {ACTIVE_ASSET} ({SYMBOL})")
print(f"  TP: {TP_PCT*100}%  |  SL: {SL_PCT*100}%  |  Lookahead: {LOOKAHEAD}")
print(f"  Session Filter: {'ON' if USE_SESSION_FILTER else 'OFF (testing mode)'}")
print(f"  Strategy: VWAP + Order Block + ICT")
print(f"{'='*55}\n")

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
        df = add_ema(df)                          # adds VWAP + EMA + RSI
        df["ema_distance"] = df["ema9"] - df["ema15"]
        df["volatility"]   = df["close"].rolling(10).std()

        # ── ICT PATTERNS ──────────────────────────────────
        try:
            df = detect_order_blocks(df, impulse_candles=3)
            df = detect_fractal_sweep(df, lookback=10)
            df = detect_cisd(df)
        except Exception as e:
            pass   # continue without ICT if it fails

        # ── SMART LABELING ────────────────────────────────
        df = label_candles(df, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead=LOOKAHEAD)

        df = df.dropna(subset=["ema9", "ema15", "ema_distance", "volatility"])
        df = df[df["label"] != -1].reset_index(drop=True)

        if df.empty:
            print("⚠️  Not enough candles yet. Waiting...")
            time.sleep(60)
            continue

        # ── SIGNAL ────────────────────────────────────────
        signal = generate_signal(
            df,
            asset              = ACTIVE_ASSET,
            use_session_filter = USE_SESSION_FILTER
        )

        last     = df.iloc[-1]
        ema_dist = float(last["ema_distance"])
        vol      = float(last["volatility"])
        label    = int(last["label"])
        price    = float(last["close"])
        vwap     = float(last["vwap"]) if "vwap" in df.columns else 0

        # ── CHECK OPEN TRADE ──────────────────────────────
        check_trade(
            current_high  = float(last["high"]),
            current_low   = float(last["low"]),
            current_price = price,
            asset         = ACTIVE_ASSET
        )

        # ── AI FILTER + EXECUTION ─────────────────────────
        if signal in ["BUY", "SELL"]:
            allow, prob = ai_filter(
                ema9         = float(last["ema9"]),
                ema15        = float(last["ema15"]),
                ema_distance = ema_dist,
                volatility   = vol,
                asset        = ACTIVE_ASSET
            )

            if allow:
                # 🔔 Sound alert — your Mac will make a sound
                send_alert(
                    f"{signal} {ACTIVE_ASSET} @ {price:,.2f} | AI: {prob:.0%}",
                    sound="Glass" if signal == "BUY" else "Basso"
                )

                opened, reason = open_trade(
                    asset  = ACTIVE_ASSET,
                    signal = signal,
                    price  = price,
                    tp_pct = TP_PCT,
                    sl_pct = SL_PCT,
                    prob   = prob
                )

                if not opened:
                    print(f"⏸️  Trade not opened: {reason}")
            else:
                print(f"🚫  {signal} BLOCKED — {ACTIVE_ASSET} | AI confidence too low: {prob:.0%}")

        # ── DISPLAY ───────────────────────────────────────
        stats      = get_stats()
        vwap_side  = "ABOVE" if price > vwap else "BELOW"
        ob_bull    = "🟢 IN BULL OB" if "price_in_bull_ob" in df.columns and last.get("price_in_bull_ob") else ""
        ob_bear    = "🔴 IN BEAR OB" if "price_in_bear_ob" in df.columns and last.get("price_in_bear_ob") else ""
        ob_tag     = ob_bull or ob_bear or ""

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{ACTIVE_ASSET:<6} | "
            f"Price: {price:>10,.2f} | "
            f"VWAP: {vwap:>10,.2f} ({vwap_side}) | "
            f"Signal: {signal:<8} | "
            f"Label: {'✅' if label == 1 else '❌'} | "
            f"Bal: ${stats['balance']:,.2f} | "
            f"WR: {stats['win_rate']}% "
            f"{ob_tag}"
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
