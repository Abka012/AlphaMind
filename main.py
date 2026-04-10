import os
import sys
import time
import traceback
import platform
import glob

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv


# Windows check - main.py only works on Windows with MetaTrader5
if platform.system() != "Windows":
    print("=" * 60)
    print("ERROR: main.py only works on Windows!")
    print("=" * 60)
    print("This script requires MetaTrader5 which is only available on Windows.")
    print("Download MetaTrader5 from: https://www.mql5.com/")
    print("")
    print("For backtesting and training, use:")
    print("  - python train.py   (train models)")
    print("  - python backtest.py (backtest models)")
    sys.exit(1)

# Check if MT5 is available
try:
    import MetaTrader5 as mt5
except ImportError:
    print("=" * 60)
    print("ERROR: MetaTrader5 not installed!")
    print("=" * 60)
    print("Please download and install MetaTrader5 from: https://www.mql5.com/")
    sys.exit(1)

load_dotenv()

# Symbol mapping: lowercase filename -> MT5 uppercase symbol
SYMBOL_MAP = {
    "eurusd": "EURUSD",
    "gbpusd": "GBPUSD",
    "usdjpy": "USDJPY",
    "audusd": "AUDUSD",
    "usdcad": "USDCAD",
    "usdchf": "USDCHF",
}

# Thresholds
OPP_THRESHOLD = float(os.getenv("OPP_THRESHOLD", "0.70"))
DIR_LONG_THRESHOLD = float(os.getenv("DIR_LONG_THRESHOLD", "0.55"))
DIR_SHORT_THRESHOLD = float(os.getenv("DIR_SHORT_THRESHOLD", "0.45"))
COMBINED_CONF_THRESHOLD = float(os.getenv("COMBINED_CONF_THRESHOLD", "0.40"))

# Risk parameters - pip based
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
SL_PIPS = float(os.getenv("SL_PIPS", "10"))
TP_PIPS = float(os.getenv("TP_PIPS", "20"))


def load_all_models():
    """Load all available models from saved_models/."""
    models = {}
    model_files = glob.glob("saved_models/*_xgb_model.pkl")
    
    for f in model_files:
        symbol = os.path.basename(f).replace("_xgb_model.pkl", "")
        try:
            data = joblib.load(f)
            models[symbol] = {
                "model_opp": data["model_opp"],
                "model_dir": data["model_dir"],
                "scaler": data["scaler"],
                "features": data["features"],
                "horizon": data.get("horizon", 1000),
            }
            print(f"Loaded model: {symbol.upper()}")
        except Exception as e:
            print(f"Failed to load model {symbol}: {e}")
    
    return models


def calculate_combined_confidence(opp_pred, dir_pred):
    """
    Calculate combined confidence for trading decision.
    Long: opp × dir
    Short: opp × (1 - dir)
    """
    if dir_pred > 0.5:  # Long signal
        return opp_pred * dir_pred
    else:  # Short signal
        return opp_pred * (1 - dir_pred)


def predict(df, model_data):
    """Get predictions from a single model."""
    scaler = model_data["scaler"]
    features = model_data["features"]
    model_opp = model_data["model_opp"]
    model_dir = model_data["model_dir"]
    
    try:
        X = df[features].values[-1:]
        X = scaler.transform(X)
        
        opp = model_opp.predict_proba(X)[0, 1]
        direction = model_dir.predict_proba(X)[0, 1]
        
        return opp, direction
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0, 0.5


def get_latest_price(symbol):
    """Get latest price from MT5."""
    tick = mt5.symbol_info_tick(SYMBOL_MAP.get(symbol, symbol.upper()))
    if tick is None:
        return None, None
    return tick.bid, tick.ask


def calculate_lot_size(balance, risk_pct, sl_pips):
    """Calculate lot size based on risk parameters."""
    risk_amount = balance * risk_pct
    pip_value = 10  # Approximate
    lot = risk_amount / (sl_pips * pip_value)
    return round(max(lot, 0.01), 2)


def place_trade(direction, symbol, lot, sl_pips, tp_pips):
    """Place a trade on MT5 with pip-based stops."""
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol.upper())
    
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        print(f"Failed to get tick for {mt5_symbol}")
        return False
    
    point = mt5.symbol_info(mt5_symbol).point
    min_stop = mt5.symbol_info(mt5_symbol).trade_stops_level * point
    
    # Get pip value: 0.01 for JPY pairs, 0.0001 for others
    def get_pip_value(sym):
        jpy_pairs = ["usdjpy", "eurjpy", "gbpjpy", "audjpy", "nzdjpy", "cadjpy", "chfjpy"]
        return 0.01 if sym.lower() in jpy_pairs else 0.0001
    
    pip_val = get_pip_value(symbol)
    sl_dist = max(min_stop, sl_pips * pip_val)
    tp_dist = max(min_stop, tp_pips * pip_val)
    
    if direction == "BUY":
        price = tick.ask
        sl = price - sl_dist
        tp = price + tp_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + sl_dist
        tp = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "HFT Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result is None:
        print(f"Order failed: {mt5.last_error()}")
        return False
    
    print(f"Trade {direction} {mt5_symbol} | Lot: {lot} | Result: {result.retcode}")
    return True


def has_open_position(symbol):
    """Check if there's an open position for a symbol."""
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol.upper())
    positions = mt5.positions_get(symbol=mt5_symbol)
    return len(positions) > 0


def close_all_positions():
    """Close all open positions."""
    positions = mt5.positions_get()
    if len(positions) == 0:
        return
    
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        direction = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": direction,
            "price": tick.bid if pos.type == 0 else tick.ask,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        mt5.order_send(request)


def log(msg):
    """Simple logging function."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    """Main trading loop."""
    log("Initializing HFT Bot...")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        sys.exit(1)
    
    # Login
    login = int(os.getenv("LOGIN", "0"))
    password = os.getenv("PASSWORD", "")
    server = os.getenv("SERVER", "")
    
    if login > 0 and password and server:
        if not mt5.login(login, password, server=server):
            print(f"MT5 login failed: {mt5.last_error()}")
            sys.exit(1)
        log("Connected to MT5!")
    
    # Get account info
    account = mt5.account_info()
    if account is None:
        print("Failed to get account info")
        sys.exit(1)
    balance = account.balance
    log(f"Account balance: ${balance:.2f}")
    
    # Load all models
    models = load_all_models()
    
    if not models:
        print("No models loaded! Run train.py first.")
        sys.exit(1)
    
    log(f"Loaded {len(models)} models: {', '.join(models.keys())}")
    log(f"Thresholds - Opp: {OPP_THRESHOLD}, Dir: {DIR_LONG_THRESHOLD}/{DIR_SHORT_THRESHOLD}, Combined: {COMBINED_CONF_THRESHOLD}")
    
    # Track last prediction time to avoid spam
    last_prediction_time = {}
    
    log("Bot started! Waiting for signals...")
    
    try:
        while True:
            try:
                # Get predictions from ALL models
                signals = []
                
                for symbol, model_data in models.items():
                    # Skip if we already have a position for this symbol
                    if has_open_position(symbol):
                        continue
                    
                    # For now, simulate prediction (in real usage, would fetch live data)
                    # This is a placeholder - in production you'd fetch real tick data
                    # and apply features using indicators.py
                    
                    # For paper trading simulation, we'll use a placeholder
                    # In reality, you'd need to stream live ticks and compute features
                    opp_pred = 0.5  # Placeholder
                    dir_pred = 0.5  # Placeholder
                    
                    combined_conf = calculate_combined_confidence(opp_pred, dir_pred)
                    
                    # Check if this is a tradeable signal
                    if opp_pred > OPP_THRESHOLD:
                        if dir_pred > DIR_LONG_THRESHOLD:
                            signal_type = "LONG"
                            signals.append((symbol, combined_conf, signal_type, dir_pred))
                        elif dir_pred < DIR_SHORT_THRESHOLD:
                            signal_type = "SHORT"
                            signals.append((symbol, combined_conf, signal_type, dir_pred))
                
                # Filter signals by combined confidence threshold
                trade_signals = [s for s in signals if s[1] > COMBINED_CONF_THRESHOLD]
                
                # Sort by combined confidence (highest first)
                trade_signals.sort(key=lambda x: x[1], reverse=True)
                
                # Trade ALL qualifying signals
                if trade_signals:
                    log(f"Found {len(trade_signals)} trade signal(s):")
                    for symbol, conf, signal_type, direction in trade_signals:
                        lot = calculate_lot_size(balance, RISK_PER_TRADE, SL_PIPS)
                        
                        if signal_type == "LONG":
                            place_trade("BUY", symbol, lot, SL_PIPS, TP_PIPS)
                        else:
                            place_trade("SELL", symbol, lot, SL_PIPS, TP_PIPS)
                        
                        log(f"  {symbol.upper()}: {signal_type} | Conf: {conf:.3f}")
                
                time.sleep(15)
                
            except Exception as e:
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                log(f"Error: {e} | File: {tb.filename} | Line: {tb.lineno}")
                time.sleep(15)
    
    except KeyboardInterrupt:
        log("Bot stopped by user")
    
    finally:
        log("Closing all positions...")
        close_all_positions()
        mt5.shutdown()
        log("Shutdown complete")


if __name__ == "__main__":
    main()