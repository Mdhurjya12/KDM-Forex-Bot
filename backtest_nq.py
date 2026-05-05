# backtest_nq.py
# KDM Trading System — NQ Futures Backtester using real MNQ 1m data
# Uses: glbx-mdp3-20250430-20260429.ohlcv-1m.csv

import sys
import pandas as pd
import numpy as np
from strategy import (add_ema, generate_signal, label_candles,
                       detect_fractal_sweep, detect_cisd, detect_order_blocks)

# =========================
# CONFIG
# =========================
DATA_PATH = "glbx-mdp3-20250430-20260429.ohlcv-1m.csv"
SYMBOL_FILTER  = "MNQM5"      # use front-month MNQ contract
ASSET          = "NASDAQ"

TP_POINTS      = 10            # 10 NQ points take profit
SL_POINTS      = 10            # 10 NQ points stop loss  (1:1)
TICK_SIZE      = 0.25          # MNQ tick = $0.50 per tick
POINT_VALUE    = 2.0           # MNQ = $2 per point
INITIAL_BAL    = 50000.0       # typical prop fund starting balance
RISK_PER_TRADE = 1             # number of contracts per trade
MAX_DAILY_LOSS = 1000.0        # prop fund daily loss limit

# =========================
# LOAD NQ DATA
# =========================
def load_nq_data():
    print(f"📥 Loading NQ data from {DATA_PATH}...")

    df = pd.read_csv(DATA_PATH)

    # Filter to front-month MNQ only
    df = df[df["symbol"] == SYMBOL_FILTER].copy()
    print(f"   {len(df)} rows for {SYMBOL_FILTER}")

    # Parse timestamp
    df["time"] = pd.to_datetime(df["ts_event"], utc=True)

    # Keep only needed columns
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Convert price columns to float
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    print(f"   Date range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print(f"   Total candles: {len(df)}\n")

    return df


# =========================
# RUN BACKTEST
# =========================
def run_backtest():
    print("=" * 55)
    print("  KDM NQ Futures Backtest — MNQ 1m Data")
    print("=" * 55)

    # ── LOAD ──────────────────────────────────────────
    df = load_nq_data()

    # ── INDICATORS ────────────────────────────────────
    print("🔧 Calculating indicators...")
    df = add_ema(df)
    df["ema_distance"] = df["ema9"] - df["ema15"]
    df["volatility"]   = df["close"].rolling(10).std()

    # ── ICT CONCEPTS ──────────────────────────────────
    print("🔍 Detecting ICT patterns (sweep + CISD + OB)...")
    df = detect_fractal_sweep(df, lookback=10)
    df = detect_cisd(df)
    df = detect_order_blocks(df, impulse_candles=3)

    # ── LABELS ────────────────────────────────────────
    # Convert points to % for labeling
    mid_price  = df["close"].median()
    tp_pct     = TP_POINTS / mid_price
    sl_pct     = SL_POINTS / mid_price

    print(f"🏷️  Labeling candles (TP={TP_POINTS}pts SL={SL_POINTS}pts)...")
    df = label_candles(df, tp_pct=tp_pct, sl_pct=sl_pct, lookahead=30)

    # ── CLEAN ─────────────────────────────────────────
    df = df.dropna(subset=["ema9", "ema15", "ema50", "rsi",
                            "ema_distance", "volatility"])
    df = df[df["label"] != -1].reset_index(drop=True)

    print(f"   {len(df)} candles after cleaning\n")

    # ── SIMULATE ──────────────────────────────────────
    balance     = INITIAL_BAL
    trades      = []
    wins = losses = 0
    daily_pnl   = {}
    skipped_session  = 0
    skipped_daily_lim = 0

    for i in range(len(df)):
        signal = generate_signal(df.iloc[:i+1], asset=ASSET, use_session_filter=True)

        if signal != "BUY":
            if signal == "NO TRADE":
                row_time = df["time"].iloc[i]
                # Count session skips (approximate)
            continue

        row      = df.iloc[i]
        date_str = str(row["time"])[:10]

        # ── DAILY LOSS LIMIT (prop fund rule) ─────────
        day_loss = daily_pnl.get(date_str, 0)
        if day_loss <= -MAX_DAILY_LOSS:
            skipped_daily_lim += 1
            continue

        entry    = row["close"]
        tp_price = entry + TP_POINTS
        sl_price = entry - SL_POINTS

        if row["label"] == 1:
            pnl     = TP_POINTS * POINT_VALUE * RISK_PER_TRADE
            wins   += 1
            outcome = "WIN"
        else:
            pnl     = -SL_POINTS * POINT_VALUE * RISK_PER_TRADE
            losses += 1
            outcome = "LOSS"

        balance += pnl
        daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl

        # ICT confirmation flags
        recent        = df.iloc[max(0, i-2):i+1]
        had_sweep     = recent["sweep_bull"].any() if "sweep_bull" in df.columns else False
        had_cisd      = recent["cisd_bull"].any()  if "cisd_bull"  in df.columns else False

        trades.append({
            "time":       str(row["time"]),
            "entry":      round(entry, 2),
            "tp":         round(tp_price, 2),
            "sl":         round(sl_price, 2),
            "outcome":    outcome,
            "pnl_$":      round(pnl, 2),
            "balance":    round(balance, 2),
            "sweep":      had_sweep,
            "cisd":       had_cisd,
        })

    # ── RESULTS ───────────────────────────────────────
    total    = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    net_pnl  = balance - INITIAL_BAL
    avg_win  = TP_POINTS * POINT_VALUE
    avg_loss = SL_POINTS * POINT_VALUE

    print("=" * 55)
    print("  KDM NQ Backtest Results")
    print("=" * 55)
    print(f"  Data period      : {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}")
    print(f"  Candles analysed : {len(df)}")
    print(f"  Total trades     : {total}")
    print(f"  Wins             : {wins}  ({win_rate:.1f}%)")
    print(f"  Losses           : {losses}")
    print(f"  Skipped (daily $ limit) : {skipped_daily_lim}")
    print(f"  Avg win  per trade : ${avg_win:.2f}")
    print(f"  Avg loss per trade : ${avg_loss:.2f}")
    print(f"  Start balance    : ${INITIAL_BAL:,.2f}")
    print(f"  End balance      : ${balance:,.2f}")
    print(f"  Net PnL          : ${net_pnl:+,.2f}")
    print("=" * 55)

    if total > 0:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("backtest_nq_results.csv", index=False)
        print(f"\n  💾 Saved → backtest_nq_results.csv")

        # ── ICT STATS ─────────────────────────────────
        sweep_trades = trades_df[trades_df["sweep"] == True]
        cisd_trades  = trades_df[trades_df["cisd"]  == True]

        if len(sweep_trades) > 0:
            sw_wr = len(sweep_trades[sweep_trades["outcome"]=="WIN"]) / len(sweep_trades) * 100
            print(f"\n  ICT Filter Stats:")
            print(f"  Sweep confirmed trades : {len(sweep_trades)}  ({sw_wr:.1f}% win rate)")

        if len(cisd_trades) > 0:
            cd_wr = len(cisd_trades[cisd_trades["outcome"]=="WIN"]) / len(cisd_trades) * 100
            print(f"  CISD confirmed trades  : {len(cisd_trades)}  ({cd_wr:.1f}% win rate)")

        # ── FIRST 5 TRADES ────────────────────────────
        print(f"\n  First 5 trades:")
        print(trades_df.head()[["time","entry","outcome","pnl_$","balance","sweep","cisd"]].to_string(index=False))

        # ── DAILY BREAKDOWN ───────────────────────────
        print(f"\n  Daily PnL summary:")
        for date, pnl in sorted(daily_pnl.items()):
            bar = "█" * int(abs(pnl) / 20) if abs(pnl) > 0 else ""
            tag = "✅" if pnl >= 0 else "❌"
            print(f"  {date}  {tag}  ${pnl:+.2f}  {bar}")

    else:
        print("\n  ⚠️  No BUY signals found.")
        print("  Check session filter — NQ trades 9:30am–4pm EST only.")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_backtest()