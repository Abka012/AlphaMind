# AlphaMind

Medium-frequency trading bot using XGBoost and raw tick data with cTrader FIX API.

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
|----------|-------------|---------|
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
3. Subscribe to live price feeds for all symbol pairs
4. Accumulate tick data in buffers (needs 50+ ticks before trading)
5. Compute features and get predictions from ALL models
6. Calculate combined confidence: `opp × dir` (long) or `opp × (1-dir)` (short)
7. Execute trades where combined confidence > threshold
8. Check for existing positions before placing new trades

### Trading Parameters (in main.py)
```python
# Thresholds for trade signals
OPP_THRESHOLD = 0.50           # Minimum opportunity probability (0.50-0.70)
DIR_LONG_THRESHOLD = 0.52      # Direction threshold for LONG (0.52-0.55)
DIR_SHORT_THRESHOLD = 0.48    # Direction threshold for SHORT (0.45-0.48)
COMBINED_CONF_THRESHOLD = 0.25 # Minimum combined confidence (0.25-0.40)

# Risk Management
RISK_PER_TRADE = 0.02          # 2% risk per trade (used for dynamic lot sizing)
SL_PIPS = 10                   # Stop loss in pips
TP_PIPS = 20                   # Take profit in pips (2:1 reward:risk)

# Lot Sizing
BASE_LOT_SIZE = 100.0          # Base lot size for all trades
CASH_BUFFER_PERCENT = 0.20     # Reserve 20% of equity as cash buffer (never trade with)
MIN_EQUITY_THRESHOLD = 10000   # Minimum equity required to trade

# Risk Management
MAX_DRAWDOWN_PERCENT = 0.05   # 5% loss from peak triggers pause for the day
MAX_CONCURRENT_POSITIONS = 2  # Maximum open positions at once
MAX_MARGIN_USAGE_PERCENT = 0.50  # Don't open new trades if margin usage > 50%

# Correlation Filter
CORRELATED_PAIRS = {
    "EURUSD": ["GBPUSD", "USDCHF"],
    "GBPUSD": ["EURUSD", "USDCHF"],
    "USDCHF": ["EURUSD", "GBPUSD"],
}
# (AUDUSD, USDCAD, USDJPY have no strong correlations)

# Reliability
API_TIMEOUT = 5                # Timeout in seconds for API calls
MAX_RECONNECT_RETRIES = 5      # Max reconnection attempts

# Margin Call Handling
MARGIN_RECOVERY_WAIT = 60      # Seconds to wait between recovery checks
MARGIN_MAX_WAIT = 600          # Max wait time (10 minutes) before exit
```

### Bot Reliability Features

The bot includes comprehensive margin call prevention and risk management:

#### Risk Management
| Feature | Description |
|---------|-------------|
| **Cash Buffer** | Reserves 20% of equity as unused cash |
| **Dynamic Position Sizing** | Lot size scales with equity (min 20% of max) |
| **Max Drawdown** | Pauses trading after 5% loss from peak equity |
| **Min Equity Check** | Stops trading if equity drops below $10,000 |

#### Position Management
| Feature | Description |
|---------|-------------|
| **Max Positions** | Maximum 2 concurrent open positions |
| **Margin Usage Check** | Skip new trades if margin usage > 50% |
| **Correlation Filter** | Avoids same-direction trades on correlated pairs |

#### Connection & Recovery
| Feature | Description |
|---------|-------------|
| **API Timeout** | 5 second timeout on all cTrader API calls |
| **Connection Check** | Validates connection before each trading cycle |
| **Exponential Backoff** | Reconnect wait time: 2^attempt + random jitter |
| **Max Retries** | 5 attempts before exiting gracefully |
| **Margin Call Detection** | Monitors position state for margin calls |
| **Auto-Resume** | Automatically resumes trading after margin recovery |

#### Connection & Recovery
| Feature | Description |
|---------|-------------|
| **API Timeout** | 5 second timeout on all cTrader API calls |
| **Connection Check** | Validates connection before each trading cycle |
| **Exponential Backoff** | Reconnect wait time: 2^attempt + random jitter |
| **Max Retries** | 5 attempts before exiting gracefully |
| **Margin Call Detection** | Monitors position state for margin calls |
| **Auto-Resume** | Automatically resumes trading after margin recovery |

**Margin Call Handling Flow:**
1. Detect margin call state from position data
2. Close all open positions immediately
3. Wait for margin recovery (60 second intervals)
4. Resume trading after recovery (clears tick buffers)
5. Exit after 10 minutes if unrecovered

Example reconnection sequence:
- Attempt 1: wait ~2s
- Attempt 2: wait ~4s
- Attempt 3: wait ~8s
- Attempt 4: wait ~16s
- Attempt 5: wait ~32s

If all retries fail, the bot logs the failure and exits cleanly rather than hanging.

### Bot Workflow

1. **Initialization**: Connect to cTrader, subscribe to symbols, load models, get initial equity
2. **Warm-up**: Wait for 50 ticks per symbol to accumulate in buffers (~12+ minutes)
3. **Trading Loop** (every 15 seconds):
   - Check connection health
   - Check for margin call state
     - If margin call: close positions, wait for recovery, auto-resume
   - Check max drawdown (pause if 5% loss from peak)
   - Check equity minimum ($10,000 threshold)
   - Check margin usage (skip if > 50%)
   - Fetch latest prices for all symbols (with timeout)
   - Update tick buffers with new price data
   - For each symbol:
     - Compute features
     - Get model predictions
     - Calculate combined confidence
     - Check thresholds
     - Check max positions (max 2)
     - Check correlation filter
   - Execute trades with dynamic lot sizing (scales with equity)

### Lot Size

The bot uses a fixed lot size for all trades (configured via `FIXED_LOT_SIZE`).

```python
FIXED_LOT_SIZE = 100.0  # Fixed lot for all trades
```

This simplifies position sizing - no need to calculate based on account balance or risk percentage.

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
- Call `api.subscribe(symbol)` before `api.quote(symbol)`

**"No price for symbol"**
- Check symbol is in your broker's offering
- Verify market is open

### Connection & Timeout Issues

**Bot freezes/hangs after running for hours**
- Fixed with timeout protection (5s) and automatic reconnection
- Bot now performs connection health check before each cycle
- Uses exponential backoff with jitter for reconnection attempts
- Logs reconnection status and exits gracefully if all retries fail

**"API call timed out" messages**
- Normal behavior when network is slow
- Bot will retry automatically with the next cycle

**Reconnection loop**
- If bot keeps reconnecting, check your network connection
- Verify cTrader server is accessible
- Consider using a more stable internet connection

### Trading Issues

**No trades being placed despite strong signals**
- Lower thresholds temporarily to test: `OPP_THRESHOLD = 0.50`, `COMBINED_CONF_THRESHOLD = 0.25`
- Check that tick buffers have reached 50+ ticks
- Review prediction values in logs

**Trades not executing / duplicate trades**
- Check cTrader account for open positions
- The position detection may fail if API returns empty symbol fields
- Manually close positions from cTrader platform if needed

**Price precision errors**
- SL/TP prices now rounded to correct decimal places (5 for most pairs, 3 for JPY)

### Margin Call Handling

**Bot detects margin call and auto-recovers**
- Bot monitors position state for margin call indicators
- On margin call: automatically closes all positions
- Waits for margin recovery (60 second intervals)
- Automatically resumes trading after recovery
- Clears tick buffers on resume for fresh data
- Exits gracefully after 10 minutes if margin not recovered

**Margin Prevention Strategies**
The bot includes multiple layers of protection:

1. **Cash Buffer**: Never trades with 20% of equity (reserved as buffer)
2. **Dynamic Lot Sizing**: Lot size scales down as equity decreases
3. **Max Drawdown**: Pauses trading for the day after 5% loss from peak
4. **Min Equity**: Stops if equity drops below $10,000
5. **Max Positions**: Limits to 2 concurrent trades
6. **Margin Usage**: Skips new trades if margin usage > 50%
7. **Correlation Filter**: Avoids same-direction trades on correlated pairs (EURUSD/GBPUSD/USDCHF)

### General

**Model errors**
- Ensure models are trained: run `python train.py` first
- Check `saved_models/` folder has `.pkl` files

### Performance Notes

- Bot checks for signals every 15 seconds
- Requires 50+ ticks in buffer before generating predictions (~12+ minutes startup)
- Each symbol accumulates ticks independently
- Use demo account for testing before live trading

## Disclaimer

This bot is for educational purposes. Past performance does not guarantee future results. Use at your own risk.
