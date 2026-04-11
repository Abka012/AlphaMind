# AlphaMind

High-frequency trading bot using XGBoost and raw tick data with cTrader FIX API.

## Overview

This trading bot uses machine learning (XGBoost) to predict price direction on multiple forex pairs using raw tick data from Dukascopy. One model is trained per trading pair.

Trading is executed via **cTrader FIX API** for low-latency execution.

## Requirements

- Python 3.10+
- cTrader FIX API credentials (get from your broker)
- Internet connection

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment (see .env Setup below)

## .env Setup

### Getting cTrader FIX Credentials

Contact your broker to get FIX API credentials. You'll need:

| Variable | Description | Example |
|----------|------------|----------|
| `CTRADER_HOST` | FIX API server host | `demo-uk-eqx-01.p.c-trader.com` |
| `CTRADER_PORT` | FIX port (SSL) | `5212` |
| `CTRADER_ACCOUNT` | SenderCompID | `demo.icmarkets.9973397` |
| `CTRADER_PASSWORD` | FIX password | (your password) |
| `CTRADER_BROKER` | Broker ID | `demo.icmarkets` |

### Create .env File

Create a `.env` file in the project root:

```bash
# cTrader FIX API Credentials
CTRADER_HOST=your_host
CTRADER_PORT=5212
CTRADER_ACCOUNT=your_sendercompid
CTRADER_PASSWORD=your_password
CTRADER_BROKER=your_broker_id
```

Example for ICMarkets demo:
```
CTRADER_HOST=demo-uk-eqx-01.p.c-trader.com
CTRADER_PORT=5212
CTRADER_ACCOUNT=demo.icmarkets.9973397
CTRADER_PASSWORD=your_password
CTRADER_BROKER=demo.icmarkets
```

### Key Points

- **Account format**: Must include broker prefix (e.g., `demo.icmarkets.9973397`)
- **Broker ID**: Get from broker (common: `demo.icmarkets`, `live.icmarkets`)
- **Host**: Different for demo vs live - check with your broker
- **Password**: Use your cTrader platform password (not Google login if linked)

## Data Download

Download tick data for all pairs for training:
```bash
python data/download_raw_ticks.py
```

This will create:
- `data/raw_ticks/eurusd.csv` - millions of tick rows
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

Train models for all 6 pairs:
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

## Live Trading

Run the trading bot:
```bash
python main.py
```

The bot will:
1. Connect to cTrader FIX API
2. Load all available models from `saved_models/`
3. Fetch live prices from cTrader
4. Compute features and get predictions from ALL models
5. Calculate combined confidence: `opp × dir` (long) or `opp × (1-dir)` (short)
6. Trade ALL pairs where combined confidence > 0.40

### Trading Parameters (in main.py)
```python
OPP_THRESHOLD = 0.70        # Minimum opportunity probability
DIR_LONG_THRESHOLD = 0.55   # Direction threshold for LONG
DIR_SHORT_THRESHOLD = 0.45 # Direction threshold for SHORT
COMBINED_CONF_THRESHOLD = 0.40  # Minimum combined confidence

RISK_PER_TRADE = 0.02      # 2% risk per trade
SL_PIPS = 10              # Stop loss in pips
TP_PIPS = 20              # Take profit in pips
```

## Model Features (25 features)

- Price: tick_ma_10, tick_ma_50, tick_ma_100, tick_ma_200
- Volatility: tick_std, tick_std_50
- Momentum: tick_momentum_10, tick_momentum_50, tick_momentum_100
- Trend: tick_trend, tick_trend_50, tick_trend_normalized
- Volume: tick_volume_ratio, tick_volume_spike
- Spread: tick_spread_pct
- Indicators: tick_rsi_centered, tick_atr
- Regime: tick_vol_regime, tick_trend_regime
- Time: hour, london_session, ny_session, asian_session, overlap_session

## Data Source

- **Training**: Dukascopy Bank SA (free tick data)
- **Trading**: Your broker via cTrader FIX API

## Data Structure

```
AlphaMind/
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
├── train.py               # Training script
├── backtest.py           # Backtesting script
├── main.py               # Live trading bot
├── indicators.py         # Feature engineering
├── requirements.txt
├── .env                 # Your credentials (not in git)
└── README.md
```

## Supported Pairs

1. EURUSD - Euro/US Dollar
2. GBPUSD - British Pound/US Dollar
3. USDJPY - US Dollar/Japanese Yen
4. AUDUSD - Australian Dollar/US Dollar
5. USDCAD - US Dollar/Canadian Dollar
6. USDCHF - US Dollar/Swiss Franc

## Troubleshooting

### Connection Issues

**"Failed to connect to cTrader: list index out of range"**
- Check `CTRADER_ACCOUNT` format - must be `broker.account` (e.g., `demo.icmarkets.9973397`)
- Verify credentials are correct

**"Ctrader.__init__() got an unexpected keyword argument"**
- Use positional args: `Ctrader(host, account, password)` not keyword args

**"Not connected to cTrader FIX API"**
- Verify `CTRADER_HOST` is correct (demo vs live are different)
- Wait for market open (markets closed on weekends)

### Price Issues

**"api.quote() returns empty"**
- Market may be closed (weekends)
- Demo accounts may not have weekend pricing
- Wait for market open (Monday ~5pm EST)

**"No price for symbol"**
- Check symbol is in your broker's offering
- Verify market is open

### General

**"Failed to get account info"**
- The method names vary by ejtraderCT version
- Bot uses `positions()` as fallback (defaults to $10,000 balance)

**Model errors**
- Ensure models are trained: run `python train.py` first
- Check `saved_models/` folder has `.pkl` files

## Disclaimer

This bot is for educational purposes. Past performance does not guarantee future results. Use at your own risk.