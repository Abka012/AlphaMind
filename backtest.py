import os

import joblib
import numpy as np
import pandas as pd
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
HORIZON = 1000  # ticks ahead (matches training)

# Thresholds - aligned with main.py
OPP_THRESHOLD = 0.50
DIR_LONG_THRESHOLD = 0.52
DIR_SHORT_THRESHOLD = 0.45
COMBINED_CONF_THRESHOLD = 0.40


def calculate_combined_confidence(opp_pred, dir_pred):
    return np.where(
        dir_pred > 0.5,
        opp_pred * dir_pred,
        opp_pred * (1 - dir_pred)
    )


def run_backtest(symbol):
    """Run backtest for a single symbol."""
    model_path = f"saved_models/{symbol}_xgb_model.pkl"

    if not os.path.exists(model_path):
        return None

    print(f"\n{'=' * 50}")
    print(f"Loading model for {symbol.upper()}...")
    print(f"{'=' * 50}")

    xgb_data = joblib.load(model_path)
    model_opp = xgb_data["model_opp"]
    model_dir = xgb_data["model_dir"]
    scaler = xgb_data["scaler"]
    features = xgb_data["features"]
    trained_horizon = xgb_data.get("horizon", HORIZON)

    from indicators import add_features, filter_data, standardize_df

    print(f"\nLoading raw tick data...")
    df = standardize_df(symbol)
    print(f"Loaded {len(df):,} ticks")

    print("\nAdding features...")
    df = add_features(df)
    print(f"After features: {len(df):,}")

    df = filter_data(df)
    df = df.dropna(subset=features)
    print(f"After dropna: {len(df):,}")

    print("\nMaking predictions...")
    X = scaler.transform(df[features])

    opp_pred = model_opp.predict_proba(X)[:, 1]
    dir_pred = model_dir.predict_proba(X)[:, 1]

    print(f"Predictions complete: {len(opp_pred):,} samples")

    # Backtest parameters - pip-based
    sl_pips = 10
    tp_pips = 20
    cost = 0.00005

    # Get pip value: 0.01 for JPY pairs, 0.0001 for others
    def get_pip_value(sym):
        jpy_pairs = [
            "usdjpy",
            "eurjpy",
            "gbpjpy",
            "audjpy",
            "nzdjpy",
            "cadjpy",
            "chfjpy",
        ]
        return 0.01 if sym.lower() in jpy_pairs else 0.0001

    pip_value = get_pip_value(symbol)

    # Calculate combined confidence
    combined_conf = calculate_combined_confidence(opp_pred, dir_pred)

    # Apply thresholds - aligned with main.py
    trade_mask = opp_pred > OPP_THRESHOLD
    positions = np.zeros(len(dir_pred))
    positions[
        trade_mask
        & (dir_pred > DIR_LONG_THRESHOLD)
        & (combined_conf > COMBINED_CONF_THRESHOLD)
    ] = 1
    positions[
        trade_mask
        & (dir_pred < DIR_SHORT_THRESHOLD)
        & (combined_conf > COMBINED_CONF_THRESHOLD)
    ] = -1

    close_prices = df["close"].values

    trade_indices = np.where(positions != 0)[0]
    valid_indices = trade_indices[trade_indices + trained_horizon < len(close_prices)]
    print(f"\nTotal positions: {len(trade_indices)}, Valid: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("No trades to analyze!")
        return {"symbol": symbol, "trades": 0, "win_rate": 0, "profit_factor": 0}

    returns = []

    for idx in tqdm(valid_indices, desc=f"Backtesting {symbol}", leave=False):
        entry_price = close_prices[idx]
        sl_distance = sl_pips * pip_value
        tp_distance = tp_pips * pip_value
        future_prices = close_prices[idx + 1 : idx + trained_horizon + 1]

        if positions[idx] == 1:  # Long
            tp_hit = np.where(future_prices >= entry_price + tp_distance)[0]
            sl_hit = np.where(future_prices <= entry_price - sl_distance)[0]

            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                ret = tp_distance / entry_price
            elif len(sl_hit) > 0:
                ret = -sl_distance / entry_price
            else:
                ret = (future_prices[-1] - entry_price) / entry_price
        else:  # Short
            tp_hit = np.where(future_prices <= entry_price - tp_distance)[0]
            sl_hit = np.where(future_prices >= entry_price + sl_distance)[0]

            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                ret = tp_distance / entry_price
            elif len(sl_hit) > 0:
                ret = -sl_distance / entry_price
            else:
                ret = (entry_price - future_prices[-1]) / entry_price

        returns.append(ret)

    returns = np.array(returns) - cost
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = total_wins / total_losses

    equity = np.cumprod(1 + returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 60 * 60)
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    max_drawdown = 1 - (equity / np.maximum.accumulate(equity)).min()

    return {
        "symbol": symbol,
        "trades": len(returns),
        "win_rate": win_rate,
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": equity[-1],
    }


if __name__ == "__main__":
    print(f"\n{'#' * 50}")
    print(f"# BACKTESTING ALL MODELS")
    print(
        f"# Horizon: {HORIZON} ticks | Opp: {OPP_THRESHOLD}, Dir: {DIR_LONG_THRESHOLD}/{DIR_SHORT_THRESHOLD}, Combined: {COMBINED_CONF_THRESHOLD}"
    )
    print(f"{'#' * 50}")

    all_results = []

    for symbol in SYMBOLS:
        result = run_backtest(symbol)

        if result is not None:
            all_results.append(result)

            print(f"\n--- {symbol.upper()} Results ---")
            print(f"  Trades: {result['trades']}")
            print(f"  Win Rate: {result['win_rate']:.3f}")
            print(f"  Profit Factor: {result['profit_factor']:.3f}")
            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  Max DD: {result['max_drawdown']:.3f}")
            print(f"  Total Return: {result['total_return']:.3f}")
        else:
            print(f"\n⚠ Skipping {symbol.upper()}: no model found")

    print("\n" + "=" * 50)
    print("AGGREGATE SUMMARY")
    print("=" * 50)
    print(
        f"{'Symbol':<10} {'Trades':>8} {'Win%':>8} {'PF':>8} {'Sharpe':>10} {'DD':>8}"
    )
    print("-" * 50)

    total_trades = 0
    for r in all_results:
        print(
            f"{r['symbol'].upper():<10} {r['trades']:>8} {r['win_rate'] * 100:>7.1f}% {r['profit_factor']:>8.2f} {r['sharpe']:>10.2f} {r['max_drawdown']:>8.3f}"
        )
        total_trades += r["trades"]

    print("-" * 50)
    print(f"{'TOTAL':<10} {total_trades:>8}")
    print("=" * 50)
    print(f"✅ Backtest complete!")

    # Save performance data for dynamic symbol selection
    performance_data = {}
    for r in all_results:
        performance_data[r['symbol']] = {
            "win_rate": r.get("win_rate", 0),
            "profit_factor": r.get("profit_factor", 0),
            "trades": r.get("trades", 0)
        }
    
    import json
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/performance.json", 'w') as f:
        json.dump(performance_data, f)
    
    print(f"\nPerformance data saved to saved_models/performance.json")
