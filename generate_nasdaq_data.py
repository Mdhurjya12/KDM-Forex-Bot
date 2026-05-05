# generate_nasdaq_data.py
# KDM Trading System — Generate NASDAQ training data from historical NQ CSV
# Converts glbx-mdp3-20250430-20260429.ohlcv-1m.csv into ai_data_nasdaq.csv

import pandas as pd
import numpy as np
from strategy import add_ema, label_candles, detect_fractal_sweep, detect_cisd

# =========================
# CONFIG
# =========================
CSV_FILE    = "glbx-mdp3-20250430-20260429.ohlcv-1m.csv"
OUTPUT_FILE = "ai_data_nasdaq.csv"
SYMBOL      = "MNQM5"       # front month contract

TP_PCT      = 0.0003        # 0.03% — ~6 NQ points
SL_PCT      = 0.0003        # 1:1 ratio
LOOKAHEAD   = 40            # 40 candles to hit TP/SL

# How many rows to sample (keeps it balanced and fast)
MAX_ROWS    = 2000

# =========================
# LOAD DATA
# =========================
def load_data():
    print(f"📥 Loading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    # Filter to front month symbol
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        print(f"   Symbols found: {list(symbols)}")

        if SYMBOL in symbols:
            df = df[df["symbol"] == SYMBOL].copy()
            print(f"   Using {SYMBOL}: {len(df)} rows")
        else:
            # Use whichever symbol has most rows
            best = df["symbol"].value_counts().index[0]
            df   = df[df["symbol"] == best].copy()
            print(f"   {SYMBOL} not found, using {best}: {len(df)} rows")

    # Parse timestamp
    df["time"] = pd.to_datetime(df["ts_event"], utc=True)

    # Keep needed columns
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Convert to float
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    print(f"   Date range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"   Total rows: {len(df)}\n")

    return df


# =========================
# GENERATE FEATURES
# =========================
def generate_features(df):
    print("🔧 Calculating indicators...")

    df = add_ema(df)
    df["ema_distance"] = df["ema9"]  - df["ema15"]
    df["volatility"]   = df["close"].rolling(10).std()

    print("🔍 Detecting ICT patterns...")
    try:
        df = detect_fractal_sweep(df, lookback=10)
        df = detect_cisd(df)
    except Exception as e:
        print(f"   ICT detection skipped: {e}")

    print("🏷️  Labeling candles...")
    df = label_candles(df, tp_pct=TP_PCT, sl_pct=SL_PCT, lookahead=LOOKAHEAD)

    # Clean
    df = df.dropna(subset=["ema9", "ema15", "ema_distance", "volatility"])
    df = df[df["label"] != -1].reset_index(drop=True)

    print(f"   {len(df)} labeled candles\n")
    return df


# =========================
# BALANCE DATASET
# =========================
def balance_and_sample(df):
    wins   = df[df["label"] == 1]
    losses = df[df["label"] == 0]

    print(f"📊 Raw labels:")
    print(f"   WIN  (1): {len(wins)}")
    print(f"   LOSS (0): {len(losses)}")

    if len(wins) == 0:
        print("\n❌ Still no WIN labels found!")
        print("   TP target may still be too high for this data.")
        print("   Try lowering TP_PCT further in this script.")
        return None

    # Sample to balance classes (max MAX_ROWS total)
    per_class = min(len(wins), len(losses), MAX_ROWS // 2)
    wins_s    = wins.sample(n=per_class,   random_state=42)
    losses_s  = losses.sample(n=per_class, random_state=42)

    balanced  = pd.concat([wins_s, losses_s]).sample(frac=1, random_state=42)
    balanced  = balanced.reset_index(drop=True)

    print(f"\n✅ Balanced dataset:")
    print(f"   WIN  (1): {len(wins_s)}")
    print(f"   LOSS (0): {len(losses_s)}")
    print(f"   Total   : {len(balanced)}\n")

    return balanced


# =========================
# SAVE
# =========================
def save_data(df):
    cols = ["ema9", "ema15", "ema_distance", "volatility", "label"]
    df[cols].to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved {len(df)} rows → {OUTPUT_FILE}")


# =========================
# MAIN
# =========================
def main():
    print("\n🚀 KDM — NASDAQ Training Data Generator")
    print("=" * 50)

    # Load
    df = load_data()

    # Features + labels
    df = generate_features(df)

    # Check label distribution
    label_counts = df["label"].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")

    if len(label_counts) < 2:
        print("\n❌ Only one label class. Trying lower TP...")
        print("   Edit TP_PCT in this script and run again.")
        return

    # Balance and sample
    df_balanced = balance_and_sample(df)
    if df_balanced is None:
        return

    # Save
    save_data(df_balanced)

    print(f"\n✅ Done! Now run:")
    print(f"   python3 train_ai.py NASDAQ")


if __name__ == "__main__":
    main()
