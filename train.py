import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SYMBOLS = [
    "eurusd",
    "gbpusd",
    "usdjpy",
    "audusd",
    "usdcad",
    "usdchf",
    "nzusd",
    "eurjpy",
    "gbpjpy",
    "audjpy",
    "eurgbp",
]
HORIZONS = [50, 100, 200]


def print_target_stats(horizon, y1, y2):
    print(f"\nTarget Statistics (Horizon {horizon}):")
    print(
        f"  Opportunity - Mean: {y1.mean():.3f}, Std: {y1.std():.3f}, Sum: {y1.sum():.0f}"
    )
    print(
        f"  Direction - Mean: {y2.mean():.3f}, Std: {y2.std():.3f}, Sum: {y2.sum():.0f}"
    )


def train_symbol(symbol):
    from indicators import add_features, add_targets, filter_data, standardize_df

    print(f"\n{'=' * 50}")
    print(f"Loading raw tick data for {symbol.upper()}...")
    print(f"{'=' * 50}")

    df = standardize_df(symbol)
    print(f"After standardize: {len(df):,} ticks")

    print("\nAdding features...")
    df = add_features(df)
    print(f"After add_features: {len(df):,}")

    df = filter_data(df)
    print(f"After filter_data: {len(df):,}")

    df = add_targets(df, horizons=HORIZONS)
    print(f"After add_targets: {len(df):,}")

    # Sample for faster training
    if len(df) > MAX_SAMPLES:
        df = df.iloc[-MAX_SAMPLES:]
        print(f"Sampled to: {len(df):,}")

    # Tick-based features + micro-structure for HFT
    features = [
        # Core features
        "tick_ma_10",
        "tick_ma_50",
        "tick_ma_100",
        "tick_ma_200",
        "tick_std",
        "tick_std_50",
        "tick_momentum_10",
        "tick_momentum_50",
        "tick_momentum_100",
        "tick_trend",
        "tick_trend_50",
        "tick_trend_normalized",
        "tick_position",
        "tick_volume_ratio",
        "tick_volume_spike",
        "tick_spread_pct",
        "tick_rsi_centered",
        "tick_atr",
        "tick_vol_regime",
        "tick_trend_regime",
        "hour",
        "london_session",
        "ny_session",
        "asian_session",
        "overlap_session",
        # Micro-structure features for HFT
        "tick_direction_imbalance",
        "spread_compression",
        "tick_acceleration",
        "vwap_deviation",
        "volume_weighted_imbalance",
        "consecutive_bid",
        "consecutive_ask",
        # Additional HFT features
        "tick_spread",
        "tick_position_normalized",
        # New Order Flow features
        "cum_delta",
        "cum_delta_deviation",
        "trade_intensity",
    ]

    print(f"\nUsing {len(features)} features")
    print(f"Training on {len(df):,} samples for {symbol.upper()}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    models = {}
    summary = {}

    for horizon in HORIZONS:
        print(f"\n{'-' * 30}")
        print(f"Training for Horizon: {horizon}")
        print(f"{'-' * 30}")

        y_opp = df[f"target_opportunity_{horizon}"].values
        y_dir = df[f"target_direction_{horizon}"].values

        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_opp_train, y_opp_test = y_opp[:split], y_opp[split:]
        y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

        print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")
        print_target_stats(horizon, y_opp_train, y_dir_train)

        # Train Opportunity model
        print(f"\nTraining XGBoost for Opportunity (H{horizon})...")
        pos_weight_opp = (len(y_opp_train) - y_opp_train.sum()) / (
            y_opp_train.sum() + 1
        )
        model_opp = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for faster ensemble training
            max_depth=6,
            learning_rate=0.03,
            min_child_weight=10,
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=3,
            scale_pos_weight=pos_weight_opp,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )

        model_opp.fit(
            X_train, y_opp_train, eval_set=[(X_test, y_opp_test)], verbose=False
        )

        opp_test_pred = model_opp.predict_proba(X_test)[:, 1]
        opp_test_acc = ((opp_test_pred > 0.5) == y_opp_test).mean()

        # Train Direction model
        print(f"Training XGBoost for Direction (H{horizon})...")
        pos_weight_dir = (len(y_dir_train) - y_dir_train.sum()) / (
            y_dir_train.sum() + 1
        )
        model_dir = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.03,
            min_child_weight=10,
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=3,
            scale_pos_weight=pos_weight_dir,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )

        model_dir.fit(
            X_train, y_dir_train, eval_set=[(X_test, y_dir_test)], verbose=False
        )

        dir_test_pred = model_dir.predict_proba(X_test)[:, 1]
        dir_test_acc = ((dir_test_pred > 0.5) == y_dir_test).mean()

        print(f"H{horizon} Opportunity Acc: {opp_test_acc:.3f}")
        print(f"H{horizon} Direction Acc: {dir_test_acc:.3f}")

        models[horizon] = {"model_opp": model_opp, "model_dir": model_dir}
        summary[horizon] = {"opp_acc": opp_test_acc, "dir_acc": dir_test_acc}

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(
        {
            "models": models,
            "scaler": scaler,
            "features": features,
            "horizons": HORIZONS,
        },
        f"saved_models/{symbol}_xgb_ensemble_model.pkl",
    )

    print(f"\n{'=' * 50}")
    print(f"✅ {symbol.upper()} XGBoost Ensemble model saved!")
    print(f"{'=' * 50}")

    return {"symbol": symbol, "summary": summary}


MAX_SAMPLES = 1000000  # Limit samples for faster training


if __name__ == "__main__":
    print(f"\n{'#' * 50}")
    print(f"# Training Ensemble models for {len(SYMBOLS)} currency pairs")
    print(f"# Horizons: {HORIZONS} ticks")
    print(f"# Max samples per symbol: {MAX_SAMPLES:,}")
    print(f"{'#' * 50}")

    results = []
    # All symbols from the SYMBOLS list
    target_symbols = SYMBOLS

    for idx, symbol in enumerate(target_symbols):
        print(f"\n[{(idx + 1)}/{len(target_symbols)}] Processing {symbol.upper()}...")
        if not os.path.exists(f"data/raw_ticks/{symbol}.csv"):
            print(f"  ⚠ Data file not found: data/raw_ticks/{symbol}.csv")
            print(f"  Run: python data/download_raw_ticks.py")
            continue

        result = train_symbol(symbol)
        results.append(result)

        print(f"\n  Completed {symbol.upper()}")
        print("-" * 40)

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    for r in results:
        msg = f"  {r['symbol'].upper()}: "
        for h, stats in r["summary"].items():
            msg += f"H{h}(Opp={stats['opp_acc']:.3f}, Dir={stats['dir_acc']:.3f}) "
        print(msg)
    print("=" * 50)
    print(f"✅ All models trained and saved to saved_models/")
