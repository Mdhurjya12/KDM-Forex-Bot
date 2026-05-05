# strategy.py
# KDM Trading System — Indicators + Signals + ICT Concepts
# Includes: EMA, RSI, Session Filter, Fractal Sweep, CISD

import pandas as pd
import numpy as np
from datetime import time as dtime

# =========================
# ADD EMA INDICATORS
# =========================
def add_ema(df):
    df = df.copy()

    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"]   = calculate_rsi(df["close"], period=14)

    return df


# =========================
# RSI
# =========================
def calculate_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =========================
# SESSION TIMING FILTER
# =========================
# ICT concept: only trade during high-probability sessions
# London Open  : 02:00 – 05:00 EST  (best for GOLD)
# NY Open      : 09:30 – 11:30 EST  (best for NASDAQ/NQ)
# NY Afternoon : 13:30 – 16:00 EST  (secondary window)
# Avoid        : 12:00 – 13:30 EST  (lunch chop)
# Avoid        : 20:00 – 02:00 EST  (dead zone)

SESSION_WINDOWS = {
    "BTC": [
        (dtime(2,  0), dtime(5,  0)),    # London open
        (dtime(9, 30), dtime(11, 30)),   # NY open
        (dtime(13,30), dtime(16,  0)),   # NY afternoon
    ],
    "GOLD": [
        (dtime(2,  0), dtime(5,  0)),    # London open — best for Gold
        (dtime(9, 30), dtime(11, 30)),   # NY open
    ],
    "NASDAQ": [
        (dtime(9, 30), dtime(11, 30)),   # NY open — primary NQ window
        (dtime(13,30), dtime(15, 30)),   # NY afternoon
    ],
}

def is_valid_session(timestamp, asset="BTC"):
    """
    Returns True if the candle timestamp falls inside a
    high-probability trading session for the given asset.
    Timestamps are converted to EST (UTC-5, no DST adjustment here).
    """
    try:
        if hasattr(timestamp, 'tz_localize'):
            ts = timestamp
        else:
            ts = pd.Timestamp(timestamp)

        # Convert to EST (UTC-5)
        if ts.tzinfo is not None:
            ts_est = ts.tz_convert("America/New_York")
        else:
            ts_est = ts - pd.Timedelta(hours=5)

        t = ts_est.time()
    except Exception:
        return True   # if timestamp parsing fails, allow trade

    windows = SESSION_WINDOWS.get(asset, SESSION_WINDOWS["BTC"])
    for start, end in windows:
        if start <= t <= end:
            return True

    return False


# =========================
# FRACTAL SWEEP DETECTION
# =========================
# ICT concept: price sweeps a prior swing high/low (takes liquidity)
# then reverses — this is a high-probability entry trigger.
#
# Bullish sweep : price wicks BELOW a prior swing low then closes above it
# Bearish sweep : price wicks ABOVE a prior swing high then closes below it

def detect_fractal_sweep(df, lookback=10):
    """
    Adds columns:
        sweep_bull  : True if current candle swept a prior low and reversed
        sweep_bear  : True if current candle swept a prior high and reversed
        sweep_level : the price level that was swept
    """
    df = df.copy()
    df["sweep_bull"]  = False
    df["sweep_bear"]  = False
    df["sweep_level"] = np.nan

    for i in range(lookback + 1, len(df)):
        window = df.iloc[i - lookback : i]

        prior_low  = window["low"].min()
        prior_high = window["high"].max()

        curr_low   = df["low"].iloc[i]
        curr_high  = df["high"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_open  = df["open"].iloc[i]

        # Bullish sweep: wick below prior low but close ABOVE it
        if curr_low < prior_low and curr_close > prior_low:
            df.at[df.index[i], "sweep_bull"]  = True
            df.at[df.index[i], "sweep_level"] = prior_low

        # Bearish sweep: wick above prior high but close BELOW it
        elif curr_high > prior_high and curr_close < prior_high:
            df.at[df.index[i], "sweep_bear"]  = True
            df.at[df.index[i], "sweep_level"] = prior_high

    return df


# =========================
# CISD DETECTION (Python)
# =========================
# Change in State of Delivery — translated from the PineScript indicator.
# Bullish CISD : prior candle was bearish, current candle is bullish
#                AND current close breaks above the prior candle's open
# Bearish CISD : prior candle was bullish, current candle is bearish
#                AND current close breaks below the prior candle's open

def detect_cisd(df):
    """
    Adds columns:
        cisd_bull  : bullish change in delivery
        cisd_bear  : bearish change in delivery
        cisd_level : the open price of the prior candle (the key level)
    """
    df = df.copy()
    df["cisd_bull"]  = False
    df["cisd_bear"]  = False
    df["cisd_level"] = np.nan

    for i in range(2, len(df)):
        prev_open  = df["open"].iloc[i - 1]
        prev_close = df["close"].iloc[i - 1]
        curr_open  = df["open"].iloc[i]
        curr_close = df["close"].iloc[i]

        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open

        # Bullish CISD: prior candle bearish, current bullish,
        # current close breaks ABOVE prior candle open
        if prev_bearish and curr_bullish and curr_close > prev_open:
            df.at[df.index[i], "cisd_bull"]  = True
            df.at[df.index[i], "cisd_level"] = prev_open

        # Bearish CISD: prior candle bullish, current bearish,
        # current close breaks BELOW prior candle open
        elif prev_bullish and curr_bearish and curr_close < prev_open:
            df.at[df.index[i], "cisd_bear"]  = True
            df.at[df.index[i], "cisd_level"] = prev_open

    return df


# =========================
# ORDER BLOCK DETECTION
# =========================
# ICT concept: last bearish candle before a bullish impulse (bull OB)
# or last bullish candle before a bearish impulse (bear OB).
# Price often returns to these zones for high-probability entries.

def detect_order_blocks(df, impulse_candles=3):
    """
    Adds columns:
        ob_bull      : True if this candle is a bullish order block
        ob_bear      : True if this candle is a bearish order block
        ob_bull_high : top of bullish OB zone
        ob_bull_low  : bottom of bullish OB zone
        ob_bear_high : top of bearish OB zone
        ob_bear_low  : bottom of bearish OB zone
    """
    df = df.copy()
    df["ob_bull"]      = False
    df["ob_bear"]      = False
    df["ob_bull_high"] = np.nan
    df["ob_bull_low"]  = np.nan
    df["ob_bear_high"] = np.nan
    df["ob_bear_low"]  = np.nan

    for i in range(impulse_candles + 1, len(df)):
        # Check for bullish impulse move after this candle
        future = df.iloc[i + 1 : i + 1 + impulse_candles]
        if len(future) < impulse_candles:
            continue

        curr_bearish = df["close"].iloc[i] < df["open"].iloc[i]
        curr_bullish = df["close"].iloc[i] > df["open"].iloc[i]

        # Bull OB: bearish candle followed by strong bullish impulse
        if curr_bearish:
            impulse_up = all(
                future["close"].iloc[j] > future["open"].iloc[j]
                for j in range(len(future))
            )
            if impulse_up:
                df.at[df.index[i], "ob_bull"]      = True
                df.at[df.index[i], "ob_bull_high"] = df["open"].iloc[i]
                df.at[df.index[i], "ob_bull_low"]  = df["close"].iloc[i]

        # Bear OB: bullish candle followed by strong bearish impulse
        if curr_bullish:
            impulse_down = all(
                future["close"].iloc[j] < future["open"].iloc[j]
                for j in range(len(future))
            )
            if impulse_down:
                df.at[df.index[i], "ob_bear"]      = True
                df.at[df.index[i], "ob_bear_high"] = df["close"].iloc[i]
                df.at[df.index[i], "ob_bear_low"]  = df["open"].iloc[i]

    return df


# =========================
# GENERATE SIGNAL (full ICT filter)
# =========================
def generate_signal(df, asset="BTC", use_session_filter=True):
    """
    Full signal with EMA crossover + RSI + trend + session + sweep + CISD.
    Returns: "BUY", "SELL", or "NO TRADE"
    """
    if len(df) < 50:
        return "NO TRADE"

    # ── SESSION FILTER ────────────────────────────────
    if use_session_filter:
        try:
            last_time = df["time"].iloc[-1] if "time" in df.columns else df.index[-1]
            if not is_valid_session(last_time, asset):
                return "NO TRADE"
        except Exception:
            pass   # if time column missing, skip session filter

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── EMA CROSSOVER ─────────────────────────────────
    ema_cross_up   = prev["ema9"] < prev["ema15"] and last["ema9"] > last["ema15"]
    ema_cross_down = prev["ema9"] > prev["ema15"] and last["ema9"] < last["ema15"]

    # ── TREND FILTER ──────────────────────────────────
    above_trend = last["close"] > last["ema50"]
    below_trend = last["close"] < last["ema50"]

    # ── RSI FILTER ────────────────────────────────────
    rsi_buy_ok  = 45 < last["rsi"] < 65
    rsi_sell_ok = 35 < last["rsi"] < 55

    # ── EMA GAP (no noise crossovers) ─────────────────
    gap_pct = abs(last["ema9"] - last["ema15"]) / last["close"]
    gap_ok  = gap_pct > 0.0002

    # ── ICT CONFIRMATIONS ─────────────────────────────
    # Check last 3 candles for sweep/CISD confirmation
    recent = df.tail(3)

    has_bull_sweep = "sweep_bull" in df.columns and recent["sweep_bull"].any()
    has_bear_sweep = "sweep_bear" in df.columns and recent["sweep_bear"].any()
    has_cisd_bull  = "cisd_bull"  in df.columns and recent["cisd_bull"].any()
    has_cisd_bear  = "cisd_bear"  in df.columns and recent["cisd_bear"].any()

    # ICT confirmation bonus — if present, relax RSI slightly
    ict_bull_confirm = has_bull_sweep or has_cisd_bull
    ict_bear_confirm = has_bear_sweep or has_cisd_bear

    if ict_bull_confirm:
        rsi_buy_ok = 40 < last["rsi"] < 70
    if ict_bear_confirm:
        rsi_sell_ok = 30 < last["rsi"] < 60

    # ── BUY ───────────────────────────────────────────
    if ema_cross_up and above_trend and rsi_buy_ok and gap_ok:
        return "BUY"

    # ── SELL ──────────────────────────────────────────
    if ema_cross_down and below_trend and rsi_sell_ok and gap_ok:
        return "SELL"

    return "NO TRADE"


# =========================
# SMART LABELING
# =========================
def label_candles(df, tp_pct=0.002, sl_pct=0.002, lookahead=30):
    """
    Forward-looking labels based on TP/SL outcome.
    label = 1  → TP hit first (win)
    label = 0  → SL hit first or neither (loss/skip)
    label = -1 → not enough future candles (drop these rows)
    """
    labels = []

    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(-1)
            continue

        entry  = df["close"].iloc[i]
        tp     = entry * (1 + tp_pct)
        sl     = entry * (1 - sl_pct)
        result = 0

        for j in range(1, lookahead + 1):
            future_high = df["high"].iloc[i + j]
            future_low  = df["low"].iloc[i + j]

            if future_high >= tp:
                result = 1
                break
            if future_low <= sl:
                result = 0
                break

        labels.append(result)

    df["label"] = labels
    return df