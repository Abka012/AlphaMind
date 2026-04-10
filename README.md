# Brandon-bot

High-frequency trading bot using XGBoost and raw tick data.

## Overview

This trading bot uses machine learning (XGBoost) to predict price direction on multiple forex pairs using raw tick data from Dukascopy. One model is trained per trading pair.

## Requirements

- Python 3.10+
- For live/paper trading on Windows: MetaTrader5 (download from https://www.mql5.com/)

### Installation

1. Install dependencies:
   ```bash
   pip install pandas numpy xgboost scikit-learn joblib tqdm dukascopy-python python-dotenv
   ```

2. (Windows only) For paper trading: Download and install MetaTrader5 from https://www.mql5.com/

## Data Download

Download 3 months of tick data for all pairs:
```bash
python data/download_raw_ticks.py
```

This will create:
- `data/raw_ticks/eurusd.csv` - ~millions of tick rows
- `data/raw_ticks/gbpusd.csv`
- `data/raw_ticks/usdjpy.csv`
- `data/raw_ticks/audusd.csv`
- `data/raw_ticks/usdcad.csv`
- `data/raw_ticks/usdchf.csv`

**Data format (tick-level):**
- `timestamp` - UTC datetime
- `bidPrice` - Bid price
- `askPrice` - Ask price
- `bidVolume` - Bid volume
- `askVolume` - Ask volume

## Training

Train models for all 6 pairs automatically:
```bash
python train.py
```

Models are saved to:
- `saved_models/eurusd_xgb_model.pkl`
- `saved_models/gbpusd_xgb_model.pkl`
- etc.

## Backtesting

Run backtest for all available models:
```bash
python backtest.py
```

This will backtest all models in `saved_models/` and print a summary table.

### Backtest Parameters
- Horizon: 1000 ticks (~3-15 minutes)
- Stop Loss: 10 pips
- Take Profit: 20 pips (2:1 RR)
- Opportunity threshold: 0.70
- Direction thresholds: 0.55 (long) / 0.45 (short)

## Live Trading (Windows Only)

**Note:** `main.py` only works on Windows with MetaTrader5 installed.

For paper trading:
```bash
python main.py
```

The bot will:
1. Load all available models from `saved_models/`
2. Get predictions from ALL models for incoming market data
3. Calculate combined confidence: `opp × dir` (long) or `opp × (1-dir)` (short)
4. Trade ALL pairs where combined confidence > 0.40

## Configuration

### Training Parameters
- Timeframe: Raw tick data
- Prediction horizon: 1000 ticks (~3-15 minutes depending on volatility)
- Stop Loss: 10 pips
- Take Profit: 20 pips (2:1 RR)
- XGBoost: n_estimators=200, max_depth=6, learning_rate=0.03

### Model Features (25 features)
- Price: tick_ma_10, tick_ma_50, tick_ma_100, tick_ma_200
- Volatility: tick_std, tick_std_50
- Momentum: tick_momentum_10, tick_momentum_50, tick_momentum_100
- Trend: tick_trend, tick_trend_50, tick_trend_normalized
- Volume: tick_volume_ratio, tick_volume_spike
- Spread: tick_spread_pct
- Indicators: tick_rsi_centered, tick_atr
- Regime: tick_vol_regime, tick_trend_regime
- Time: hour, london_session, ny_session, asian_session, overlap_session

### Data Source
- **Dukascopy Bank SA** - Free tick data feed
- 3 months historical data
- Full tick precision (bid/ask prices)

## Data Structure

```
Brandon-bot/
├── data/
│   ├── raw_ticks/
│   │   ├── eurusd.csv      # Raw tick data
│   │   ├── gbpusd.csv
│   │   └── ...
│   └── download_raw_ticks.py  # Data download script
├── saved_models/
│   ├── eurusd_xgb_model.pkl
│   ├── gbpusd_xgb_model.pkl
│   └── ...
├── train.py               # Training script (trains all pairs)
├── backtest.py           # Backtesting script (tests all models)
├── main.py               # Live trading (Windows + MT5 only)
├── indicators.py         # Feature engineering
├── cleanup.sh            # Cleanup old data/models
├── requirements.txt
└── README.md
```

## Supported Pairs

1. EURUSD - Euro/US Dollar
2. GBPUSD - British Pound/US Dollar
3. USDJPY - US Dollar/Japanese Yen
4. AUDUSD - Australian Dollar/US Dollar
5. USDCAD - US Dollar/Canadian Dollar
6. USDCHF - US Dollar/Swiss Franc

## Disclaimer

This bot is for educational purposes. Past performance does not guarantee future results. Use at your own risk.