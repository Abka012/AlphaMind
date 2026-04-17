import numpy as np
import pandas as pd
from tqdm import tqdm


def standardize_df(symbol="eurusd"):
    """Load raw tick data from data/raw_ticks folder."""
    df = pd.read_csv(f"data/raw_ticks/{symbol}.csv", sep="\t")
    
    df.columns = ["timestamp", "bidPrice", "askPrice", "bidVolume", "askVolume"]
    
    # Calculate mid price (close)
    df["close"] = (df["bidPrice"] + df["askPrice"]) / 2
    
    # Total volume
    df["tick_volume"] = df["bidVolume"] + df["askVolume"]
    
    # Handle timezone-aware timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    
    # Skip first 1000 ticks for indicator warmup
    df = df.iloc[1000:]
    
    return df


def add_features(df):
    """Compute features on rolling tick windows."""
    print("Computing tick features...")
    
    df = df.copy()
    
    # Price-based features with tqdm progress
    print("  - Price MAs...")
    for window in [10, 50, 100, 200]:
        df[f'tick_ma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
    
    print("  - Volatility...")
    df['tick_std'] = df['close'].rolling(window=100, min_periods=1).std()
    df['tick_std_50'] = df['close'].rolling(window=50, min_periods=1).std()
    
    print("  - Momentum...")
    for shift in [10, 50, 100]:
        df[f'tick_momentum_{shift}'] = df['close'] - df['close'].shift(shift)
    
    print("  - Trend features...")
    df['tick_trend'] = df['tick_ma_10'] - df['tick_ma_100']
    df['tick_trend_50'] = df['tick_ma_10'] - df['tick_ma_50']
    df['tick_trend_normalized'] = df['tick_trend'] / (df['tick_std'] + 1e-9)
    
    print("  - Price position...")
    df['tick_high_100'] = df['close'].rolling(window=100, min_periods=1).max()
    df['tick_low_100'] = df['close'].rolling(window=100, min_periods=1).min()
    df['tick_position'] = (df['close'] - df['tick_low_100']) / (df['tick_high_100'] - df['tick_low_100'] + 1e-9)
    df['tick_position_normalized'] = (df['close'] - df['tick_ma_50']) / (df['tick_std'] + 1e-9)
    
    print("  - Volume features...")
    df['tick_volume_ma'] = df['tick_volume'].rolling(window=50, min_periods=1).mean()
    df['tick_volume_ratio'] = df['tick_volume'] / (df['tick_volume_ma'] + 1e-9)
    df['tick_volume_spike'] = (df['tick_volume'] > df['tick_volume_ma'] * 2).astype(int)
    
    print("  - Spread features...")
    df['tick_spread'] = df['askPrice'] - df['bidPrice']
    df['tick_spread_pct'] = df['tick_spread'] / df['close']
    df['tick_spread_ma'] = df['tick_spread'].rolling(window=50, min_periods=1).mean()
    
    print("  - RSI...")
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df['tick_rsi'] = 100 - (100 / (1 + rs))
    df['tick_rsi_centered'] = df['tick_rsi'] - 50
    
    print("  - Time features...")
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    
    # Session flags
    df['london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    # ATR-like feature for stops
    df['tick_true_range'] = df['close'].diff().abs()
    df['tick_atr'] = df['tick_true_range'].ewm(span=100, adjust=False).mean()
    
    # Regime features
    df['tick_vol_regime'] = (df['tick_std'] > df['tick_std'].rolling(200, min_periods=1).mean()).astype(int)
    df['tick_trend_regime'] = (df['tick_trend'] > 0).astype(int)

    # Micro-structure features for HFT
    df['tick_direction'] = np.sign(df['close'].diff())
    df['tick_direction'] = df['tick_direction'].replace(0, np.nan).ffill().fillna(0)
    
    df['tick_direction_imbalance'] = (
        df['tick_direction'].rolling(window=20, min_periods=1).sum() / 20
    )
    
    df['spread_ma'] = df['tick_spread'].rolling(window=20, min_periods=1).mean()
    df['spread_compression'] = (df['tick_spread'] < df['spread_ma'] * 0.5).astype(int)
    
    df['tick_acceleration'] = df['close'].diff().diff()
    
    df['vwap'] = (df['close'] * df['tick_volume']).rolling(window=50, min_periods=1).sum() / (
        df['tick_volume'].rolling(window=50, min_periods=1).sum() + 1e-9
    )
    df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['tick_std'] + 1e-9)
    
    df['volume_weighted_imbalance'] = (
        (df['bidVolume'] - df['askVolume']).rolling(window=20, min_periods=1).sum() / 20
    )
    
    df['consecutive_bid'] = (
        df['close'].diff().apply(lambda x: 1 if x > 0 else 0).rolling(window=5, min_periods=1).sum()
    )
    df['consecutive_ask'] = (
        df['close'].diff().apply(lambda x: 1 if x < 0 else 0).rolling(window=5, min_periods=1).sum()
    )
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    return df


def filter_data(df):
    """Filter data - minimal filtering for raw ticks."""
    return df.copy()


def add_targets(df, horizon=200):
    """Predict: will price move profitably in next N ticks?"""
    print(f"Computing targets (horizon={horizon} ticks)...")
    
    df = df.copy()
    
    # Future return after horizon ticks
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    df['target_return'] = future_return
    
    # Direction: price goes up = 1, down = 0
    df['target_direction'] = (future_return > 0).astype(int)
    
    # Opportunity: significant move threshold
    min_profit = df['tick_std'].rolling(100, min_periods=1).mean() * 1.5
    df['target_opportunity'] = (future_return.abs() > min_profit).astype(int)
    
    df = df.dropna(subset=['target_opportunity', 'target_direction'])
    
    return df
    
    return df
    
    return df


def smooth_labels(y, smoothing=0.1):
    y = np.clip(y, 0, 1)
    return y * (1 - smoothing) + 0.5 * smoothing


def update_features(df):
    return df


def append_new_candle(df, new_df):
    """Append new tick data to existing dataframe."""
    last_time = df.index[-1]
    if new_df.index[-1] > last_time:
        df = pd.concat([df, new_df])
    return df


def compute_rsi(series, period=14):
    """Compute RSI for compatibility."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))