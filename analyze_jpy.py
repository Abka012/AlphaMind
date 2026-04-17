import pandas as pd
import numpy as np
from indicators import standardize_df, add_features, add_targets

symbol = "eurusd"
df = standardize_df(symbol)
df = add_features(df)
df = add_targets(df, horizons=[50, 100, 200])

print(f"\nAnalysis for {symbol.upper()}")
print(f"Total ticks: {len(df):,}")

for h in [50, 100, 200]:
    opp_mean = df[f'target_opportunity_{h}'].mean()
    dir_mean = df[f'target_direction_{h}'].mean()
    print(f"H{h} - Opp Mean: {opp_mean:.4f}, Dir Mean: {dir_mean:.4f}")

# Check volatility
print(f"Mean tick_std: {df['tick_std'].mean():.6f}")
print(f"Mean tick_spread: {df['tick_spread'].mean():.6f}")
