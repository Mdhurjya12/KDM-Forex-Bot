# strategy.py
# KDM Trading System — VWAP + Order Block + ICT Concepts
# Upgraded per Kapil: EMA replaced with VWAP, Order Block entries added

import pandas as pd
import numpy as np
from datetime import time as dtime

# =========================
# ADD INDICATORS
# EMA kept for labeling/AI features
# VWAP added as primary signal indicator
# =========================
def add_ema(df):
    """
    Adds VWAP, EMA (for AI features), RSI.
    Function name kept as add_ema for backward compatibility
    with main.py and train_ai.py imports.
    """
    df = df.copy()

    # ── VWAP (primary signal indicator) ───────────────
    # VWAP = cumulative(price * volume) / cumulative(volume)
    # Resets every session (daily)
    df = calculate_vwap(df)

    # ── EMA (kept for AI feature columns) ─────────────
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────
    df["rsi"] = calculate_rsi(df["close"], period=14)

    # ── VWAP DISTANCE (used as AI feature) ────────────
    df["ema_distance"] = df["close"] - df["vwap"]   # price vs VWAP

    return df


# =========================
# VWAP CALCULATION
# =========================
def calculate_vwap(df):
    """
    Calculates VWAP and VWAP bands.
    Resets daily (uses date from timestamp if available).

    Adds columns:
        vwap        : main VWAP line
        vwap_upper  : VWAP + 1 standard deviation (resistance)
        vwap_lower  : VWAP - 1 standard deviation (support)
        vwap_upper2 : VWAP + 2 standard deviations (extended resistance)
        vwap_lower2 : VWAP - 2 standard deviations (extended support)
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    # Try to reset VWAP by day
    if "time" in df.columns:
        try:
            df["_date"] = pd.to_datetime(df["time"]).dt.date
        except Exception:
            df["_date"] = 0
    else:
        df["_date"] = 0

    vwap_vals    = []
    vwap_upper   = []
    vwap_lower   = []
    vwap_upper2  = []
    vwap_lower2  = []

    cum_tp_vol   = 0.0
    cum_vol      = 0.0
    cum_tp2_vol  = 0.0
    prev_date    = None

    for i in range(len(df)):
        curr_date = df["_date"].iloc[i]

        # Reset at start of each new day
        if curr_date != prev_date:
            cum_tp_vol  = 0.0
            cum_vol     = 0.0
            cum_tp2_vol = 0.0
            prev_date   = curr_date

        tp  = typical_price.iloc[i]
        vol = df["volume"].iloc[i]
        if vol == 0:
            vol = 1

        cum_tp_vol  += tp * vol
        cum_vol     += vol
        cum_tp2_vol += (tp ** 2) * vol

        vwap = cum_tp_vol / cum_vol
        variance = max(0, (cum_tp2_vol / cum_vol) - (vwap ** 2))
        std  = variance ** 0.5

        vwap_vals.append(vwap)
        vwap_upper.append(vwap + std)
        vwap_lower.append(vwap - std)
        vwap_upper2.append(vwap + 2 * std)
        vwap_lower2.append(vwap - 2 * std)

    df["vwap"]       = vwap_vals
    df["vwap_upper"] = vwap_upper
    df["vwap_lower"] = vwap_lower
    df["vwap_upper2"]= vwap_upper2
    df["vwap_lower2"]= vwap_lower2

    if "_date" in df.columns:
        df = df.drop(columns=["_date"])

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
SESSION_WINDOWS = {
    "BTC": [
        (dtime(2,  0), dtime(5,  0)),
        (dtime(9, 30), dtime(11, 30)),
        (dtime(13,30), dtime(16,  0)),
    ],
    "GOLD": [
        (dtime(2,  0), dtime(5,  0)),
        (dtime(9, 30), dtime(11, 30)),
    ],
    "NASDAQ": [
        (dtime(9, 30), dtime(11, 30)),
        (dtime(13,30), dtime(15, 30)),
    ],
}

def is_valid_session(timestamp, asset="BTC"):
    try:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is not None:
            ts_est = ts.tz_convert("America/New_York")
        else:
            ts_est = ts - pd.Timedelta(hours=5)
        t = ts_est.time()
    except Exception:
        return True

    windows = SESSION_WINDOWS.get(asset, SESSION_WINDOWS["BTC"])
    for start, end in windows:
        if start <= t <= end:
            return True
    return False


# =========================
# ORDER BLOCK DETECTION
# =========================
def detect_order_blocks(df, impulse_candles=3):
    """
    Detects bullish and bearish order blocks.

    Bull OB : last bearish candle before a bullish impulse move
              → price returns here to buy
    Bear OB : last bullish candle before a bearish impulse move
              → price returns here to sell

    Adds columns:
        ob_bull      : True if this candle is a bullish OB
        ob_bear      : True if this candle is a bearish OB
        ob_bull_high : top of bull OB zone
        ob_bull_low  : bottom of bull OB zone
        ob_bear_high : top of bear OB zone
        ob_bear_low  : bottom of bear OB zone
        price_in_bull_ob : True if current price is inside any recent bull OB
        price_in_bear_ob : True if current price is inside any recent bear OB
    """
    df = df.copy()
    df["ob_bull"]          = False
    df["ob_bear"]          = False
    df["ob_bull_high"]     = np.nan
    df["ob_bull_low"]      = np.nan
    df["ob_bear_high"]     = np.nan
    df["ob_bear_low"]      = np.nan
    df["price_in_bull_ob"] = False
    df["price_in_bear_ob"] = False

    bull_obs = []   # list of (high, low) for active bull OBs
    bear_obs = []   # list of (high, low) for active bear OBs

    for i in range(impulse_candles + 1, len(df)):
        future = df.iloc[i + 1 : i + 1 + impulse_candles]
        if len(future) < impulse_candles:
            continue

        curr_bearish = df["close"].iloc[i] < df["open"].iloc[i]
        curr_bullish = df["close"].iloc[i] > df["open"].iloc[i]

        # Bull OB: bearish candle → strong bullish impulse after
        if curr_bearish:
            impulse_up = all(
                future["close"].iloc[j] > future["open"].iloc[j]
                for j in range(len(future))
            )
            if impulse_up:
                ob_high = df["open"].iloc[i]
                ob_low  = df["close"].iloc[i]
                df.at[df.index[i], "ob_bull"]      = True
                df.at[df.index[i], "ob_bull_high"] = ob_high
                df.at[df.index[i], "ob_bull_low"]  = ob_low
                bull_obs.append((ob_high, ob_low))

        # Bear OB: bullish candle → strong bearish impulse after
        if curr_bullish:
            impulse_down = all(
                future["close"].iloc[j] < future["open"].iloc[j]
                for j in range(len(future))
            )
            if impulse_down:
                ob_high = df["close"].iloc[i]
                ob_low  = df["open"].iloc[i]
                df.at[df.index[i], "ob_bear"]      = True
                df.at[df.index[i], "ob_bear_high"] = ob_high
                df.at[df.index[i], "ob_bear_low"]  = ob_low
                bear_obs.append((ob_high, ob_low))

        # Check if current price is inside any recent OB (last 20)
        curr_close = df["close"].iloc[i]
        recent_bull_obs = bull_obs[-20:]
        recent_bear_obs = bear_obs[-20:]

        in_bull = any(ob_low <= curr_close <= ob_high for ob_high, ob_low in recent_bull_obs)
        in_bear = any(ob_low <= curr_close <= ob_high for ob_high, ob_low in recent_bear_obs)

        df.at[df.index[i], "price_in_bull_ob"] = in_bull
        df.at[df.index[i], "price_in_bear_ob"] = in_bear

    return df


# =========================
# FRACTAL SWEEP DETECTION
# =========================
def detect_fractal_sweep(df, lookback=10):
    df = df.copy()
    df["sweep_bull"]  = False
    df["sweep_bear"]  = False
    df["sweep_level"] = np.nan

    for i in range(lookback + 1, len(df)):
        window     = df.iloc[i - lookback : i]
        prior_low  = window["low"].min()
        prior_high = window["high"].max()
        curr_low   = df["low"].iloc[i]
        curr_high  = df["high"].iloc[i]
        curr_close = df["close"].iloc[i]

        if curr_low < prior_low and curr_close > prior_low:
            df.at[df.index[i], "sweep_bull"]  = True
            df.at[df.index[i], "sweep_level"] = prior_low
        elif curr_high > prior_high and curr_close < prior_high:
            df.at[df.index[i], "sweep_bear"]  = True
            df.at[df.index[i], "sweep_level"] = prior_high

    return df


# =========================
# CISD DETECTION
# =========================
def detect_cisd(df):
    df = df.copy()
    df["cisd_bull"]  = False
    df["cisd_bear"]  = False
    df["cisd_level"] = np.nan

    for i in range(2, len(df)):
        prev_open  = df["open"].iloc[i - 1]
        prev_close = df["close"].iloc[i - 1]
        curr_open  = df["open"].iloc[i]
        curr_close = df["close"].iloc[i]

        prev_bearish = prev_close < prev_open
        prev_bullish = prev_close > prev_open
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open

        if prev_bearish and curr_bullish and curr_close > prev_open:
            df.at[df.index[i], "cisd_bull"]  = True
            df.at[df.index[i], "cisd_level"] = prev_open
        elif prev_bullish and curr_bearish and curr_close < prev_open:
            df.at[df.index[i], "cisd_bear"]  = True
            df.at[df.index[i], "cisd_level"] = prev_open

    return df


# =========================
# GENERATE SIGNAL
# Primary: VWAP + Order Block
# Confirmation: RSI + Session + Sweep + CISD
# =========================
def generate_signal(df, asset="BTC", use_session_filter=True):
    """
    VWAP + Order Block signal logic:

    BUY  conditions:
      1. Price crosses above VWAP  (momentum entry)
      OR
      2. Price pulls back into Bull Order Block near VWAP  (OB entry)
      + RSI confirmation
      + Session filter

    SELL conditions:
      1. Price crosses below VWAP
      OR
      2. Price pulls back into Bear Order Block near VWAP
      + RSI confirmation
      + Session filter
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
            pass

    last = df.iloc[-1]
    prev = df.iloc[-2]

    curr_close = last["close"]
    curr_rsi   = last["rsi"]
    curr_vwap  = last["vwap"]

    # ── VWAP CROSS ────────────────────────────────────
    # Bullish cross: price was below VWAP, now above
    vwap_cross_up   = prev["close"] < prev["vwap"] and curr_close > curr_vwap
    # Bearish cross: price was above VWAP, now below
    vwap_cross_down = prev["close"] > prev["vwap"] and curr_close < curr_vwap

    # ── VWAP POSITION ─────────────────────────────────
    above_vwap = curr_close > curr_vwap
    below_vwap = curr_close < curr_vwap

    # ── ORDER BLOCK CHECK ─────────────────────────────
    in_bull_ob = "price_in_bull_ob" in df.columns and last["price_in_bull_ob"]
    in_bear_ob = "price_in_bear_ob" in df.columns and last["price_in_bear_ob"]

    # ── RSI FILTER ────────────────────────────────────
    rsi_buy_ok  = 35 < curr_rsi < 75
    rsi_sell_ok = 25 < curr_rsi < 65

    # ── ICT CONFIRMATIONS ─────────────────────────────
    recent         = df.tail(3)
    has_bull_sweep = "sweep_bull" in df.columns and recent["sweep_bull"].any()
    has_bear_sweep = "sweep_bear" in df.columns and recent["sweep_bear"].any()
    has_cisd_bull  = "cisd_bull"  in df.columns and recent["cisd_bull"].any()
    has_cisd_bear  = "cisd_bear"  in df.columns and recent["cisd_bear"].any()

    ict_bull = has_bull_sweep or has_cisd_bull
    ict_bear = has_bear_sweep or has_cisd_bear

    if ict_bull:
        rsi_buy_ok = 30 < curr_rsi < 80
    if ict_bear:
        rsi_sell_ok = 20 < curr_rsi < 70

    # ── BUY SIGNALS ───────────────────────────────────

    # Signal 1: VWAP crossover up
    if vwap_cross_up and rsi_buy_ok:
        return "BUY"

    # Signal 2: Price in Bull Order Block + above VWAP + ICT confirm
    if in_bull_ob and above_vwap and rsi_buy_ok and ict_bull:
        return "BUY"

    # Signal 3: Bull OB + VWAP support (price near VWAP from above)
    vwap_dist_pct = (curr_close - curr_vwap) / curr_vwap
    near_vwap     = abs(vwap_dist_pct) < 0.001     # within 0.1% of VWAP
    if in_bull_ob and near_vwap and rsi_buy_ok:
        return "BUY"

    # ── SELL SIGNALS ──────────────────────────────────

    # Signal 1: VWAP crossover down
    if vwap_cross_down and rsi_sell_ok:
        return "SELL"

    # Signal 2: Price in Bear Order Block + below VWAP + ICT confirm
    if in_bear_ob and below_vwap and rsi_sell_ok and ict_bear:
        return "SELL"

    # Signal 3: Bear OB + VWAP resistance (price near VWAP from below)
    if in_bear_ob and near_vwap and rsi_sell_ok:
        return "SELL"

    return "NO TRADE"


# =========================
# SMART LABELING
# =========================
def label_candles(df, tp_pct=0.002, sl_pct=0.002, lookahead=30):
    """
    Forward-looking labels based on TP/SL outcome.
    label = 1  → TP hit first (win)
    label = 0  → SL hit first or neither (loss)
    label = -1 → not enough future candles (drop)
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
