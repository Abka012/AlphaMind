# AlphaMind

A high-frequency trading bot using XGBoost to predict price movements on forex pairs.

## What It Does

- **Predicts** price direction using tick-level data
- **Trades** automatically via cTrader FIX API
- **Optimizes** for short-term opportunities (30s-10min horizons)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your credentials
cp .env.example .env
# Edit .env with your cTrader FIX API credentials

# 3. Download training data
python data/download_raw_ticks.py

# 4. Train models
python train.py

# 5. Backtest
python backtest.py

# 6. Go live (demo first!)
python main.py
```

## Supported Pairs

11 major and cross pairs including EURUSD, GBPUSD, USDJPY, AUDUSD, and more.

## Key Features

- **Ensemble of Horizons**: Three models (50/100/200 ticks ahead) with consensus logic
- **Volatility Filter**: Only trades during high-volatility periods
- **Risk Management**: Cash buffers, max positions, correlation filters, auto-recovery
- **Reliability**: Automatic reconnection, timeout protection, margin call handling

## Requirements

- Python 3.10+
- cTrader FIX API credentials (from your broker)

## Disclaimer

Educational use only. Past performance ≠ future results. Use at your own risk.
