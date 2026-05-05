# train_ai.py
# KDM Trading System — Multi-Asset AI Training Pipeline

import sys
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     accuracy_score)

# =========================
# ASSET CONFIG
# =========================
ASSETS = {
    "BTC": {
        "data_file":   "ai_data_btc.csv",
        "model_file":  "kdm_model_btc.pkl",
        "scaler_file": "kdm_scaler_btc.pkl",
    },
    "GOLD": {
        "data_file":   "ai_data_gold.csv",
        "model_file":  "kdm_model_gold.pkl",
        "scaler_file": "kdm_scaler_gold.pkl",
    },
    "NASDAQ": {
        "data_file":   "ai_data_nasdaq.csv",
        "model_file":  "kdm_model_nasdaq.pkl",
        "scaler_file": "kdm_scaler_nasdaq.pkl",
    },
}

# ── SWITCH ASSET FROM TERMINAL ────────────────────────
# Usage:
#   python3 train_ai.py          ← defaults to BTC
#   python3 train_ai.py GOLD
#   python3 train_ai.py NASDAQ

if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in ASSETS:
        ACTIVE_ASSET = arg
    else:
        print(f"❌  Unknown asset '{arg}'. Choose: {list(ASSETS.keys())}")
        sys.exit(1)
else:
    ACTIVE_ASSET = "BTC"

CONFIG       = ASSETS[ACTIVE_ASSET]
DATA_FILE    = CONFIG["data_file"]
MODEL_FILE   = CONFIG["model_file"]
SCALER_FILE  = CONFIG["scaler_file"]

MIN_ROWS     = 100
TEST_SIZE    = 0.2
RANDOM_STATE = 42

FEATURES = ["ema9", "ema15", "ema_distance", "volatility"]
TARGET   = "label"

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
# LOAD DATA
# =========================
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌  {DATA_FILE} not found.")
        print(f"    Run: python3 main.py {ACTIVE_ASSET}")
        return None

    df = pd.read_csv(DATA_FILE)

    print(f"📂  Loaded {len(df)} rows from {DATA_FILE}")
    print(f"    Columns : {list(df.columns)}")
    print(f"    Labels  : {df[TARGET].value_counts().to_dict()}\n")

    return df


# =========================
# VALIDATE DATA
# =========================
def validate_data(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"❌  Missing columns: {missing}")
        return False

    if len(df) < MIN_ROWS:
        print(f"❌  Only {len(df)} rows — need at least {MIN_ROWS}.")
        return False

    counts = df[TARGET].value_counts()

    # ── CRITICAL CHECK — need both classes ────────────
    if len(counts) < 2:
        print(f"❌  Only one label class found: {counts.to_dict()}")
        print(f"    All rows are labeled {counts.index[0]}.")
        print(f"    This means TP is never being hit.")
        print(f"    Fix: lower tp_pct or increase LOOKAHEAD in main.py")
        print(f"    Then delete {DATA_FILE} and recollect data.")
        return False

    minority = counts.min()
    majority = counts.max()
    ratio    = minority / majority

    print(f"⚖️   Class balance  [{ACTIVE_ASSET}]:")
    print(f"    WIN  (1): {counts.get(1, 0)} rows")
    print(f"    LOSS (0): {counts.get(0, 0)} rows")
    print(f"    Ratio   : {ratio:.2f}  ", end="")
    print("⚠️  Imbalanced" if ratio < 0.3 else "✅  OK")
    print()

    return True


# =========================
# FEATURE ENGINEERING
# =========================
def engineer_features(df):
    df = df.copy()

    df["ema_ratio"]      = df["ema9"] / df["ema15"]
    df["vol_normalised"] = df["volatility"] / df["ema15"]
    df["dist_positive"]  = (df["ema_distance"] > 0).astype(int)
    df["dist_strength"]  = df["ema_distance"].abs() / df["ema15"]

    return df


# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter     = 1000,
            class_weight = "balanced",
            random_state = RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators = 200,
            max_depth    = 6,
            class_weight = "balanced",
            random_state = RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            random_state  = RANDOM_STATE
        ),
    }

    results = {}

    print("=" * 50)
    print(f"  Model Comparison — {ACTIVE_ASSET}")
    print("=" * 50)

    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="accuracy"
        )
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "model":    model,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "test_acc": test_acc,
            "y_pred":   y_pred,
        }

        print(f"\n  {name}")
        print(f"  {'─' * 40}")
        print(f"  CV Accuracy  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")

    return results


# =========================
# EVALUATE BEST MODEL
# =========================
def evaluate_best(results, X_test, y_test):
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best      = results[best_name]
    y_pred    = best["y_pred"]

    print(f"\n\n{'=' * 50}")
    print(f"  Best Model: {best_name}  [{ACTIVE_ASSET}]")
    print(f"{'=' * 50}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["LOSS (0)", "WIN (1)"]
    ))

    cm                 = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp_val = cm.ravel()

    print(f"  Confusion Matrix:")
    print(f"  ┌─────────────┬──────┬───────┐")
    print(f"  │             │   Predicted   │")
    print(f"  │             │ LOSS │  WIN  │")
    print(f"  ├─────────────┼──────┼───────┤")
    print(f"  │ Actual LOSS │ {tn:4d} │ {fp:5d} │")
    print(f"  │ Actual WIN  │ {fn:4d} │ {tp_val:5d} │")
    print(f"  └─────────────┴──────┴───────┘")

    precision = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0
    recall    = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0

    print(f"\n  WIN precision : {precision:.1%}")
    print(f"  WIN recall    : {recall:.1%}")

    # 1:1 ratio breakeven is 50%
    breakeven = 0.50
    print(f"\n  Profitability check (1:1 TP/SL):")
    print(f"  Break-even win rate : {breakeven:.1%}")
    print(f"  Model win precision : {precision:.1%}")

    if precision > breakeven:
        print(f"  Edge : +{precision - breakeven:.1%}  ✅  Profitable")
    else:
        print(f"  Gap  : -{breakeven - precision:.1%}  ❌  Needs more data")

    return best_name, best["model"]


# =========================
# SAVE MODEL
# =========================
def save_model(model, scaler):
    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n  💾  Model  saved → {MODEL_FILE}")
    print(f"  💾  Scaler saved → {SCALER_FILE}")


# =========================
# MAIN PIPELINE
# =========================
def train():
    print(f"\n🤖  KDM AI Training — {ACTIVE_ASSET}")
    print("=" * 50)

    df = load_data()
    if df is None:
        return

    if not validate_data(df):
        return

    df = engineer_features(df)
    df = df.dropna().reset_index(drop=True)

    X = df[ENGINEERED_FEATURES]
    y = df[TARGET]

    print(f"🔧  Features: {ENGINEERED_FEATURES}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    print(f"📊  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = train_models(X_train, X_test, y_train, y_test)

    best_name, best_model = evaluate_best(results, X_test, y_test)

    save_model(best_model, scaler)

    print(f"\n✅  Done. Best model: {best_name}")
    print(f"    Load in ai_filter.py with:")
    print(f"    model  = joblib.load('{MODEL_FILE}')\n")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    train()