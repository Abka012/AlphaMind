# AlphaMind

High-frequency trading (HFT) bot using XGBoost and raw tick data with cTrader FIX API.

## Overview

This trading bot uses machine learning (XGBoost) to predict price direction on multiple forex pairs using raw tick data from Dukascopy. One model is trained per trading pair.

Trading is executed via **cTrader FIX API** for low-latency execution.

## HFT Optimizations (2024)

The model has been optimized for high-frequency trading with the following improvements:

| Parameter | Original | HFT |
|-----------|----------|-----|
| Horizon | 1000 ticks | 50 ticks (~10-30 sec) |
| SL/TP | 10/20 pips | 5/10 pips |
| Check Interval | 15 sec | 3 sec |
| Features | 25 | 34 (+micro-structure) |

### New HFT Features
- **Tick Direction Imbalance**: Tracks consecutive bid/ask pressure
- **Spread Compression**: Identifies low-spread opportunities
- **VWAP Deviation**: Measures fair value deviation
- **Volume-Weighted Imbalance**: Order flow analysis
- **Acceleration**: Second-order momentum

## Development with AGENTS.md

This project uses `AGENTS.md` for agentic development workflow:

```bash
# Agent sessions are documented in docs/agent-sessions/
# Each session tracks: goal, files changed, commands run
opencode export  # Export current session
```

**Benefits:**
- Traces all code changes with clear commit messages
- Documents decision rationale and实验 results
- Enables reproducibility of optimization experiments
- Branch-based workflow: `agent/<task-name>` for each feature

See `AGENTS.md` for full workflow rules.

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
- `data/raw_ticks/nzusd.csv`
- `data/raw_ticks/eurjpy.csv`
- `data/raw_ticks/gbpjpy.csv`
- `data/raw_ticks/audjpy.csv`
- `data/raw_ticks/eurgbp.csv`

**Data format (tick-level):**
- `timestamp` - UTC datetime
- `bidPrice` - Bid price
- `askPrice` - Ask price
- `bidVolume` - Bid volume
- `askVolume` - Ask volume

## Training

Train models for all 11 pairs:
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
- Horizon: 50 ticks (~10-30 seconds)
- Stop Loss: 5 pips
- Take Profit: 10 pips (2:1 RR)
- Opportunity threshold: 0.55
- Direction thresholds: 0.52 (long) / 0.48 (short)
- Combined confidence threshold: 0.30

### Performance (EURUSD HFT)
- Win Rate: 54.5%
- Profit Factor: 1.33
- Sharpe: 1.06 (trade-based annualization)

## Live Trading

Run the trading bot:
```bash
python main.py
```

The bot will:
1. Connect to cTrader FIX API
2. Load performance data from `performance.json` (if exists)
3. Select top 6 performing symbols by profit_factor, or use all 11 if no data
4. Load models for selected symbols from `saved_models/`
5. Subscribe to live price feeds for selected symbol pairs
6. Accumulate tick data in buffers (needs 50+ ticks before trading)
7. Compute features and get predictions from ALL models
8. Calculate combined confidence: `opp × dir` (long) or `opp × (1-dir)` (short)
9. Execute trades where combined confidence > threshold
10. Check for existing positions before placing new trades

### Trading Parameters (in main.py)
```python
# Thresholds for trade signals (HFT optimized)
OPP_THRESHOLD = 0.55           # Minimum opportunity probability
DIR_LONG_THRESHOLD = 0.52      # Direction threshold for LONG
DIR_SHORT_THRESHOLD = 0.48    # Direction threshold for SHORT
COMBINED_CONF_THRESHOLD = 0.30 # Minimum combined confidence

# Risk Management
RISK_PER_TRADE = 0.02          # 2% risk per trade (used for dynamic lot sizing)
SL_PIPS = 5                    # Stop loss in pips (HFT tight)
TP_PIPS = 10                   # Take profit in pips (2:1 reward:risk)

# Lot Sizing
FIXED_LOT_SIZE = 100.0          # Fixed lot for all trades
CASH_BUFFER_PERCENT = 0.20     # Reserve 20% of equity as cash buffer (never trade with)
MIN_EQUITY_THRESHOLD = 10000   # Minimum equity required to trade

# Risk Management
MAX_DRAWDOWN_PERCENT = 0.05   # 5% loss from peak triggers pause for the day
MAX_CONCURRENT_POSITIONS = 4  # Maximum open positions at once
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

# Symbol Selection
TOP_N_SYMBOLS = 6             # Use top 6 performing symbols from backtest
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
| **Max Positions** | Maximum 4 concurrent open positions |
| **Margin Usage Check** | Skip new trades if margin usage > 50% |
| **Correlation Filter** | Avoids same-direction trades on correlated pairs |

### Bot Workflow

1. **Initialization**: Connect to cTrader, load performance.json, select top 6 symbols by profit_factor
2. **Subscribe**: Subscribe to price feeds for selected symbols only
3. **Load Models**: Load XGBoost models for selected symbols
4. **Warm-up**: Wait for 50 ticks per symbol to accumulate in buffers (~12+ minutes)
5. **Trading Loop** (every 15 seconds):
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
     - Check max positions (max 4)
     - Check correlation filter
   - Execute trades with fixed lot sizing

### Symbol Selection

The bot now supports **dynamic symbol selection** based on backtest performance:

- After running `python backtest.py`, performance data is saved to `performance.json`
- On startup, `main.py` loads this file and sorts symbols by `profit_factor`
- Selects the top 6 performing symbols for live trading
- Falls back to all 11 symbols if no performance data exists

This allows the bot to focus on historically best-performing pairs while maintaining diversity.

## Model Features (34 features)

### Core Features (25)
- Price: tick_ma_10, tick_ma_50, tick_ma_100, tick_ma_200
- Volatility: tick_std, tick_std_50
- Momentum: tick_momentum_10, tick_momentum_50, tick_momentum_100
- Trend: tick_trend, tick_trend_50, tick_trend_normalized
- Volume: tick_volume_ratio, tick_volume_spike
- Spread: tick_spread, tick_spread_pct
- Indicators: tick_rsi_centered, tick_atr
- Regime: tick_vol_regime, tick_trend_regime
- Time: hour, london_session, ny_session, asian_session, overlap_session

### HFT Micro-Structure Features (9)
- tick_direction_imbalance - Consecutive bid/ask pressure
- spread_compression - Low spread condition detection
- tick_acceleration - Second-order momentum
- vwap_deviation - Fair value deviation
- volume_weighted_imbalance - Order flow analysis
- consecutive_bid/ask - Mini-trend detection
- tick_position_normalized - Normalized price position

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

### Major Pairs (7)
1. EURUSD - Euro/US Dollar
2. GBPUSD - British Pound/US Dollar
3. USDJPY - US Dollar/Japanese Yen
4. AUDUSD - Australian Dollar/US Dollar
5. USDCAD - US Dollar/Canadian Dollar
6. USDCHF - US Dollar/Swiss Franc
7. NZDUSD - New Zealand Dollar/US Dollar

### Cross Pairs (4)
8. EURJPY - Euro/Japanese Yen
9. GBPJPY - British Pound/Japanese Yen
10. AUDJPY - Australian Dollar/Japanese Yen
11. EURGBP - Euro/British Pound

**Total: 11 symbols**

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

**KeyError: 'nzdusd' or similar symbol errors**
- Ensure `data/performance.json` is generated by running `python backtest.py` first
- The bot now gracefully handles symbols not in tick_buffers by checking existence first
- Falls back to all 11 symbols if performance data is not available

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
5. **Max Positions**: Limits to 4 concurrent trades
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
