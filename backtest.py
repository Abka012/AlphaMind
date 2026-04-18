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
HORIZON = 200  # ticks ahead (~1-5 min)

# Thresholds - selective for quality
OPP_THRESHOLD = 0.70
DIR_LONG_THRESHOLD = 0.55
DIR_SHORT_THRESHOLD = 0.45
COMBINED_CONF_THRESHOLD = 0.35


def calculate_combined_confidence(opp_pred, dir_pred):
    return np.where(dir_pred > 0.5, opp_pred * dir_pred, opp_pred * (1 - dir_pred))


def run_backtest(symbol):
    """Run backtest for a single symbol using Ensemble of Horizons."""
    model_path = f"saved_models/{symbol}_xgb_ensemble_model.pkl"

    if not os.path.exists(model_path):
        return None

    print(f"\n{'=' * 50}")
    print(f"Loading Ensemble model for {symbol.upper()}...")
    print(f"{'=' * 50}")

    xgb_data = joblib.load(model_path)
    ensemble_models = xgb_data["models"]
    scaler = xgb_data["scaler"]
    features = xgb_data["features"]
    horizons = xgb_data["horizons"]

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

    print("\nMaking predictions for all horizons...")
    X = scaler.transform(df[features])

    opp_preds = {}
    dir_preds = {}

    for h in horizons:
        print(f"  Predicting for H{h}...")
        model_opp = ensemble_models[h]["model_opp"]
        model_dir = ensemble_models[h]["model_dir"]
        opp_preds[h] = model_opp.predict_proba(X)[:, 1]
        dir_preds[h] = model_dir.predict_proba(X)[:, 1]

    print(f"Predictions complete: {len(X):,} samples")

    # Consensus Logic:
    # 1. All horizons must agree on direction
    # 2. Average direction confidence must be strong
    # 3. Average opportunity must be above threshold

    dir_consensus_long = np.ones(len(X), dtype=bool)
    dir_consensus_short = np.ones(len(X), dtype=bool)
    avg_opp = np.zeros(len(X))
    avg_dir = np.zeros(len(X))

    for h in horizons:
        dir_consensus_long &= dir_preds[h] > 0.5
        dir_consensus_short &= dir_preds[h] < 0.5
        avg_opp += opp_preds[h]
        avg_dir += dir_preds[h]

    avg_opp /= len(horizons)
    avg_dir /= len(horizons)

    # Dynamic threshold based on symbol characteristics
    # JPY pairs often have more "noise" opportunities
    is_jpy = any(x in symbol.lower() for x in ["jpy"])

    opp_thresh = 0.85 if is_jpy else 0.80
    conf_thresh = 0.60 if is_jpy else 0.55
    # Backtest parameters
    sl_pips = 10
    tp_pips = 30
    cost = 0.00005

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

    # Combined confidence for the ensemble
    combined_conf = np.where(avg_dir > 0.5, avg_opp * avg_dir, avg_opp * (1 - avg_dir))

    # Regime Filter: Volatility > average
    vol_regime = df["tick_vol_regime"].values

    positions = np.zeros(len(avg_dir))

    # Selective thresholds for consensus + Volatility Regime filter
    positions[
        dir_consensus_long
        & (avg_opp > opp_thresh)
        & (combined_conf > conf_thresh)
        & (vol_regime == 1)
    ] = 1

    positions[
        dir_consensus_short
        & (avg_opp > opp_thresh)
        & (combined_conf > conf_thresh)
        & (vol_regime == 1)
    ] = -1

    close_prices = df["close"].values
    trained_horizon = max(horizons)  # Use max horizon for exit if not hit SL/TP

    trade_indices = np.where(positions != 0)[0]
    valid_indices = trade_indices[trade_indices + trained_horizon < len(close_prices)]
    print(f"\nTotal positions: {len(trade_indices)}, Valid: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("No trades to analyze!")
        return {
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "total_return": 0,
        }

    returns = []

    for idx in tqdm(valid_indices, desc=f"Backtesting {symbol}", leave=False):
        entry_price = close_prices[idx]
        sl_distance = sl_pips * pip_value
        tp_distance = tp_pips * pip_value

        # Trailing stop parameters
        be_threshold = 5 * pip_value  # Move to breakeven after 5 pips profit
        trail_start = 10 * pip_value  # Start trailing after 10 pips profit
        trail_dist = 5 * pip_value  # Trail distance

        current_sl = (
            entry_price - sl_distance
            if positions[idx] == 1
            else entry_price + sl_distance
        )
        peak_profit = 0

        future_prices = close_prices[idx + 1 : idx + trained_horizon + 1]
        trade_ret = 0

        for i, price in enumerate(future_prices):
            if positions[idx] == 1:  # Long
                profit = price - entry_price

                # Move to breakeven
                if profit >= be_threshold and current_sl < entry_price:
                    current_sl = entry_price

                # Trail stop
                if profit >= trail_start:
                    new_sl = price - trail_dist
                    if new_sl > current_sl:
                        current_sl = new_sl

                # Check exit
                if price >= entry_price + tp_distance:
                    trade_ret = tp_distance / entry_price
                    break
                elif price <= current_sl:
                    trade_ret = (current_sl - entry_price) / entry_price
                    break
            else:  # Short
                profit = entry_price - price

                # Move to breakeven
                if profit >= be_threshold and current_sl > entry_price:
                    current_sl = entry_price

                # Trail stop
                if profit >= trail_start:
                    new_sl = price + trail_dist
                    if new_sl < current_sl:
                        current_sl = new_sl

                # Check exit
                if price <= entry_price - tp_distance:
                    trade_ret = tp_distance / entry_price
                    break
                elif price >= current_sl:
                    trade_ret = (entry_price - current_sl) / entry_price
                    break

            # Horizon exit
            if i == len(future_prices) - 1:
                trade_ret = (
                    (price - entry_price) / entry_price
                    if positions[idx] == 1
                    else (entry_price - price) / entry_price
                )

        returns.append(trade_ret)

    returns = np.array(returns) - cost
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = total_wins / total_losses

    equity = np.cumprod(1 + returns)
    # Annualize: using sqrt of number of trades for more realistic Sharpe
    # Each trade represents ~50 ticks (horizon), not 1 second
    n_periods_per_day = len(returns) / (HORIZON * 60)  # rough estimate
    sharpe = (
        np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(n_periods_per_day * 252)
        if n_periods_per_day > 0
        else 0
    )
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
        performance_data[r["symbol"]] = {
            "win_rate": r.get("win_rate", 0),
            "profit_factor": r.get("profit_factor", 0),
            "trades": r.get("trades", 0),
        }

    import json

    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/performance.json", "w") as f:
        json.dump(performance_data, f)

    print(f"\nPerformance data saved to saved_models/performance.json")
