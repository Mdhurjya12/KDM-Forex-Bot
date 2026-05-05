# ai_filter.py
# KDM Trading System — Multi-Asset AI Filter

import pandas as pd
import joblib
import os

# =========================
# CONFIG
# =========================
MIN_PROB = 0.60      # minimum confidence to approve a trade

ASSET_MODELS = {
    "BTC": {
        "model":  "kdm_model_btc.pkl",
        "scaler": "kdm_scaler_btc.pkl",
    },
    "GOLD": {
        "model":  "kdm_model_gold.pkl",
        "scaler": "kdm_scaler_gold.pkl",
    },
    "NASDAQ": {
        "model":  "kdm_model_nasdaq.pkl",
        "scaler": "kdm_scaler_nasdaq.pkl",
    },
}

ENGINEERED_FEATURES = [
    "ema9",
    "ema15",
    "ema_distance",
    "volatility",
    "ema_ratio",
    "vol_normalised",
    "dist_positive",
    "dist_strength",
]


# =========================
# LOAD MODEL + SCALER
# =========================
def load_model(asset="BTC"):
    config = ASSET_MODELS.get(asset)

    if not config:
        print(f"❌  Unknown asset: {asset}")
        print(f"    Available assets: {list(ASSET_MODELS.keys())}")
        return None, None

    model_path  = config["model"]
    scaler_path = config["scaler"]

    if not os.path.exists(model_path):
        print(f"⚠️  No model file found for {asset}: {model_path}")
        print(f"    Run: python3 train_ai.py  (with ACTIVE_ASSET = '{asset}')")
        return None, None

    if not os.path.exists(scaler_path):
        print(f"⚠️  No scaler file found for {asset}: {scaler_path}")
        print(f"    Run: python3 train_ai.py  (with ACTIVE_ASSET = '{asset}')")
        return None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


# =========================
# BUILD FEATURES
# (must match train_ai.py exactly)
# =========================
def build_features(ema9, ema15, ema_distance, volatility):
    ema_ratio      = ema9 / ema15
    vol_normalised = volatility / ema15
    dist_positive  = 1 if ema_distance > 0 else 0
    dist_strength  = abs(ema_distance) / ema15

    df = pd.DataFrame([{
        "ema9":          ema9,
        "ema15":         ema15,
        "ema_distance":  ema_distance,
        "volatility":    volatility,
        "ema_ratio":     ema_ratio,
        "vol_normalised":vol_normalised,
        "dist_positive": dist_positive,
        "dist_strength": dist_strength,
    }])

    return df[ENGINEERED_FEATURES]


# =========================
# AI FILTER
# =========================
def ai_filter(ema9, ema15, ema_distance, volatility, asset="BTC"):
    """
    Decides whether to allow or block a trade signal.

    Parameters:
        ema9         : float — current EMA9 value
        ema15        : float — current EMA15 value
        ema_distance : float — EMA9 minus EMA15
        volatility   : float — rolling std of close prices
        asset        : str   — "BTC", "GOLD", or "NASDAQ"

    Returns:
        allow : bool  — True = approve trade, False = block trade
        prob  : float — AI confidence score (0.0 to 1.0)
    """

    # ── LOAD MODEL ────────────────────────────────────
    model, scaler = load_model(asset)

    if model is None:
        # No model trained yet → allow all trades (fallback mode)
        print(f"⚠️  [{asset}] Running without AI filter — all trades allowed")
        return True, 1.0

    # ── BUILD + SCALE FEATURES ────────────────────────
    try:
        features        = build_features(ema9, ema15, ema_distance, volatility)
        features_scaled = scaler.transform(features)
    except Exception as e:
        print(f"⚠️  [{asset}] Feature error: {e} — allowing trade")
        return True, 1.0

    # ── PREDICT ───────────────────────────────────────
    try:
        prob  = model.predict_proba(features_scaled)[0][1]   # P(WIN)
        allow = prob >= MIN_PROB
    except Exception as e:
        print(f"⚠️  [{asset}] Prediction error: {e} — allowing trade")
        return True, 1.0

    return allow, round(float(prob), 4)


# =========================
# TEST FILTER (run directly)
# =========================
if __name__ == "__main__":
    print("🧪  Testing AI Filter\n")

    test_cases = [
        {"asset": "BTC",    "ema9": 83500.0, "ema15": 83450.0, "ema_distance":  50.0, "volatility": 120.0},
        {"asset": "GOLD",   "ema9":  2350.0, "ema15":  2348.0, "ema_distance":   2.0, "volatility":   3.5},
        {"asset": "NASDAQ", "ema9":   445.0, "ema15":   444.5, "ema_distance":   0.5, "volatility":   1.2},
    ]

    for case in test_cases:
        allow, prob = ai_filter(
            ema9         = case["ema9"],
            ema15        = case["ema15"],
            ema_distance = case["ema_distance"],
            volatility   = case["volatility"],
            asset        = case["asset"]
        )
        status = "✅  APPROVED" if allow else "🚫  BLOCKED"
        print(f"  {case['asset']:<6} → {status}  |  confidence: {prob:.0%}")