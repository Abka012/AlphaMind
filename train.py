import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SYMBOLS = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "usdchf"]
HORIZON = 1000  # ticks ahead to predict (~3-15 minutes)


def print_target_stats(y1, y2):
    print("\nTarget Statistics:")
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

    df = add_targets(df, horizon=HORIZON)
    print(f"After add_targets: {len(df):,}")

    df = df.dropna(subset=["target_opportunity", "target_direction"])
    print(f"After dropna: {len(df):,}")

    # Tick-based features
    features = [
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
    ]

    print(f"\nUsing {len(features)} features")
    print(f"Training on {len(df):,} samples for {symbol.upper()}")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    y_opp = df["target_opportunity"].values
    y_dir = df["target_direction"].values

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_opp_train, y_opp_test = y_opp[:split], y_opp[split:]
    y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

    print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")
    print_target_stats(y_opp_train, y_dir_train)

    # Train Opportunity model
    print("\n" + "=" * 50)
    print("Training XGBoost for Opportunity...")
    print("=" * 50)

    pos_weight_opp = (len(y_opp_train) - y_opp_train.sum()) / (y_opp_train.sum() + 1)
    model_opp = xgb.XGBClassifier(
        n_estimators=200,
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

    with tqdm(total=1, desc=f"{symbol.upper()} Opportunity", leave=False) as pbar:
        model_opp.fit(
            X_train, y_opp_train, eval_set=[(X_test, y_opp_test)], verbose=False
        )
        pbar.update(1)

    opp_train_pred = model_opp.predict_proba(X_train)[:, 1]
    opp_train_acc = ((opp_train_pred > 0.5) == y_opp_train).mean()
    opp_test_pred = model_opp.predict_proba(X_test)[:, 1]
    opp_test_acc = ((opp_test_pred > 0.5) == y_opp_test).mean()

    print(f"Opportunity Train Accuracy: {opp_train_acc:.3f}")
    print(f"Opportunity Test Accuracy: {opp_test_acc:.3f}")

    # Train Direction model
    print("\n" + "=" * 50)
    print("Training XGBoost for Direction...")
    print("=" * 50)

    pos_weight_dir = (len(y_dir_train) - y_dir_train.sum()) / (y_dir_train.sum() + 1)
    model_dir = xgb.XGBClassifier(
        n_estimators=200,
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

    with tqdm(total=1, desc=f"{symbol.upper()} Direction", leave=False) as pbar:
        model_dir.fit(
            X_train, y_dir_train, eval_set=[(X_test, y_dir_test)], verbose=False
        )
        pbar.update(1)

    dir_train_pred = model_dir.predict_proba(X_train)[:, 1]
    dir_train_acc = ((dir_train_pred > 0.5) == y_dir_train).mean()
    dir_test_pred = model_dir.predict_proba(X_test)[:, 1]
    dir_test_acc = ((dir_test_pred > 0.5) == y_dir_test).mean()

    print(f"Direction Train Accuracy: {dir_train_acc:.3f}")
    print(f"Direction Test Accuracy: {dir_test_acc:.3f}")

    print("\n--- Feature Importance (Direction) ---")
    importance = model_dir.feature_importances_
    feat_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f"  {feat}: {imp:.3f}")

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(
        {
            "model_opp": model_opp,
            "model_dir": model_dir,
            "scaler": scaler,
            "features": features,
            "horizon": HORIZON,
        },
        f"saved_models/{symbol}_xgb_model.pkl",
    )

    print(f"\n{'=' * 50}")
    print(f"✅ {symbol.upper()} XGBoost model saved!")
    print(f"{'=' * 50}")

    return {
        "symbol": symbol,
        "opp_test_acc": opp_test_acc,
        "dir_test_acc": dir_test_acc,
    }


if __name__ == "__main__":
    print(f"\n{'#' * 50}")
    print(f"# Training models for {len(SYMBOLS)} currency pairs")
    print(f"# Horizon: {HORIZON} ticks")
    print(f"{'#' * 50}")

    results = []
    for idx, symbol in enumerate(SYMBOLS):
        print(f"\n[{(idx + 1)}/{len(SYMBOLS)}] Processing {symbol.upper()}...")

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
        print(
            f"  {r['symbol'].upper()}: Opp={r['opp_test_acc']:.3f}, Dir={r['dir_test_acc']:.3f}"
        )
    print("=" * 50)
    print(f"✅ All models trained and saved to saved_models/")
