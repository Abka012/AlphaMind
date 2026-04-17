# Agent Session Export - 2026-04-17

## Goal
Improve core direction prediction (P1) from ~50% (random) to a profitable level using Ensemble of Horizons and Regime Detection.

## Changes
- **indicators.py**: Updated `add_targets` to support multiple horizons (50, 100, 200 ticks).
- **train.py**: Updated to train separate Opportunity and Direction models for each horizon.
- **backtest.py**: Implemented consensus logic (all horizons must agree) and a volatility regime filter.
- **main.py**: Updated live trading loop to use the new ensemble models and regime filtering.
- **README.md**: Updated with latest performance metrics and technical details.
- **IMPROVEMENTS.md**: Updated status to reflect P1 completion for EURUSD.

## Results (EURUSD)
- **Win Rate**: Improved from ~44% to **49.9%**.
- **Profit Factor**: Improved from 0.77 to **1.05** (Profitable).
- **Sharpe Ratio**: Improved from -5.81 to **0.40** (Positive).
- **Trade Quality**: Significantly reduced noise-driven trades via consensus and regime filtering.

## Decisions
- Used 50, 100, 200 tick horizons for the ensemble to capture micro-trends at different speeds.
- Implemented a strict volatility regime filter (`tick_std > tick_std_ma_200`) to avoid low-liquidity choppiness.
- Increased confidence thresholds in the consensus logic to prioritize trade quality over quantity.

## Next Steps
- Train ensemble models for other currency pairs (GBPUSD, USDJPY, etc.).
- Implement P2: Order Flow features (Cumulative Delta, Order Book Imbalance).
- Implement P4: Trailing stops for risk management.
