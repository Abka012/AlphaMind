import glob
import os
import socket
import sys
import time
import traceback
import warnings
from collections import deque

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ejtraderCT import Ctrader

load_dotenv()

CTRADER_HOST = os.getenv("CTRADER_HOST")
CTRADER_ACCOUNT = os.getenv("CTRADER_ACCOUNT")
CTRADER_PASSWORD = os.getenv("CTRADER_PASSWORD")
CTRADER_BROKER = os.getenv("CTRADER_BROKER")

SYMBOL_MAP = {
    "eurusd": "EURUSD",
    "gbpusd": "GBPUSD",
    "usdjpy": "USDJPY",
    "audusd": "AUDUSD",
    "usdcad": "USDCAD",
    "usdchf": "USDCHF",
}

TICK_HISTORY = 200
tick_buffers = {symbol: deque(maxlen=TICK_HISTORY) for symbol in SYMBOL_MAP}

OPP_THRESHOLD = 0.70
DIR_LONG_THRESHOLD = 0.55
DIR_SHORT_THRESHOLD = 0.45
COMBINED_CONF_THRESHOLD = 0.40

RISK_PER_TRADE = 0.02
SL_PIPS = 10
TP_PIPS = 20

DEFAULT_BALANCE = 1000000  # Set to your actual account balance

api = None


def check_connection():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((CTRADER_HOST, int(os.getenv("CTRADER_PORT", "5212"))))
        sock.close()
        return result == 0
    except:
        return False


def load_all_models():
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
    if dir_pred > 0.5:
        return opp_pred * dir_pred
    else:
        return opp_pred * (1 - dir_pred)


def predict(df, model_data):
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


def get_latest_prices():
    prices = {}
    for symbol in SYMBOL_MAP.values():
        try:
            q = api.quote(symbol)
            if q:
                if "bid" in q and "ask" in q:
                    prices[symbol] = q
                elif "last" in q:
                    prices[symbol] = {"bid": q["last"], "ask": q["last"]}
        except Exception as e:
            pass

    if prices:
        return prices

    try:
        positions = api.positions()
        for pos in positions:
            symbol = pos.get("symbol", "")
            if symbol:
                prices[symbol.lower()] = {
                    "bid": pos.get("bid", 0),
                    "ask": pos.get("ask", 0),
                }
        return prices
    except:
        return {}


def update_tick_buffer(symbol, bid, ask):
    ctrader_symbol = SYMBOL_MAP.get(symbol, symbol.upper())
    mid = (bid + ask) / 2
    tick_buffers[symbol].append(
        {
            "timestamp": pd.Timestamp.now(),
            "close": mid,
            "bidPrice": bid,
            "askPrice": ask,
            "tick_volume": 1,
        }
    )


def compute_live_features(symbol):
    buffer = tick_buffers[symbol]

    if len(buffer) < 50:
        return None

    df = pd.DataFrame(buffer)
    df["close"] = (df["bidPrice"] + df["askPrice"]) / 2
    df["tick_volume"] = df.get("tick_volume", pd.Series([1] * len(df)))
    df["tick_spread"] = df["askPrice"] - df["bidPrice"]
    df["hour"] = df["timestamp"].dt.hour

    features = {}
    for window in [10, 50, 100, 200]:
        features[f"tick_ma_{window}"] = (
            df["close"].rolling(window=window, min_periods=1).mean().iloc[-1]
        )

    features["tick_std"] = df["close"].rolling(window=100, min_periods=1).std().iloc[-1]
    df["tick_std"] = df["close"].rolling(window=100, min_periods=1).std()
    features["tick_std_50"] = (
        df["close"].rolling(window=50, min_periods=1).std().iloc[-1]
    )

    for shift in [10, 50, 100]:
        if len(df) > shift:
            features[f"tick_momentum_{shift}"] = (
                df["close"].iloc[-1] - df["close"].iloc[-shift - 1]
            )
        else:
            features[f"tick_momentum_{shift}"] = 0

    features["tick_trend"] = features["tick_ma_10"] - features["tick_ma_100"]
    features["tick_trend_50"] = features["tick_ma_10"] - features["tick_ma_50"]
    features["tick_trend_normalized"] = features["tick_trend"] / (
        features["tick_std"] + 1e-9
    )

    features["tick_high_100"] = (
        df["close"].rolling(window=100, min_periods=1).max().iloc[-1]
    )
    features["tick_low_100"] = (
        df["close"].rolling(window=100, min_periods=1).min().iloc[-1]
    )
    features["tick_position"] = (df["close"].iloc[-1] - features["tick_low_100"]) / (
        features["tick_high_100"] - features["tick_low_100"] + 1e-9
    )

    features["tick_volume_ma"] = (
        df["tick_volume"].rolling(window=50, min_periods=1).mean().iloc[-1]
    )
    features["tick_volume_ratio"] = df["tick_volume"].iloc[-1] / (
        features["tick_volume_ma"] + 1e-9
    )
    features["tick_volume_spike"] = 1 if features["tick_volume_ratio"] > 2 else 0

    features["tick_spread"] = df["tick_spread"].iloc[-1]
    spread_pct = features["tick_spread"] / (df["close"].iloc[-1] + 1e-9)
    features["tick_spread_pct"] = spread_pct

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean().iloc[-1]
    rs = gain / (loss + 1e-9)
    features["tick_rsi_centered"] = (100 - (100 / (1 + rs))) - 50

    features["tick_atr"] = delta.abs().ewm(span=100, adjust=False).mean().iloc[-1]
    features["tick_vol_regime"] = (
        1
        if features["tick_std"]
        > df["tick_std"].rolling(200, min_periods=1).mean().iloc[-1]
        else 0
    )
    features["tick_trend_regime"] = 1 if features["tick_trend"] > 0 else 0

    features["hour"] = df["hour"].iloc[-1]
    features["london_session"] = 1 if 7 <= features["hour"] < 16 else 0
    features["ny_session"] = 1 if 13 <= features["hour"] < 21 else 0
    features["asian_session"] = 1 if 0 <= features["hour"] < 8 else 0
    features["overlap_session"] = 1 if 13 <= features["hour"] < 16 else 0

    return features


def calculate_lot_size(balance, risk_pct, sl_pips):
    risk_amount = balance * risk_pct
    pip_value = 10
    lot = risk_amount / (sl_pips * pip_value)
    max_lot = 100.0
    return round(min(max(lot, 0.01), max_lot), 2)


def get_pip_value(symbol):
    jpy_pairs = ["usdjpy", "eurjpy", "gbpjpy", "audjpy", "nzdjpy", "cadjpy", "chfjpy"]
    return 0.01 if symbol.lower() in jpy_pairs else 0.0001


def place_trade(direction, symbol, lot, sl_pips, tp_pips):
    ctrader_symbol = SYMBOL_MAP.get(symbol, symbol.upper())

    try:
        positions = api.positions()
        for pos in positions:
            if pos.get("symbol", "").upper() == ctrader_symbol.upper():
                log(f"Position already exists for {ctrader_symbol}, skipping")
                return False
    except:
        pass

    try:
        prices = get_latest_prices()
        price_data = prices.get(ctrader_symbol, {})
        bid = price_data.get("bid", 0)
        ask = price_data.get("ask", 0)

        if not bid or not ask:
            log(f"No price for {ctrader_symbol}")
            return False

        pip_val = get_pip_value(symbol)
        digits = 3 if symbol.lower() == "usdjpy" else 5

        if direction == "BUY":
            sl = round(bid - (sl_pips * pip_val), digits)
            tp = round(bid + (tp_pips * pip_val), digits)
            api.buy(ctrader_symbol, lot, sl, tp)
            log(
                f"Trade BUY {ctrader_symbol} | Lot: {lot} | SL: {sl} | TP: {tp}"
            )
        else:
            sl = round(ask + (sl_pips * pip_val), digits)
            tp = round(ask - (tp_pips * pip_val), digits)
            api.sell(ctrader_symbol, lot, sl, tp)
            log(
                f"Trade SELL {ctrader_symbol} | Lot: {lot} | SL: {sl} | TP: {tp}"
            )

        return True

    except Exception as e:
        log(f"Trade failed: {e}")
        return False


def has_open_position(symbol):
    ctrader_symbol = SYMBOL_MAP.get(symbol, symbol.upper())
    try:
        positions = api.positions()
        for pos in positions:
            if pos.get("symbol", "").upper() == ctrader_symbol.upper():
                return True
        return False
    except:
        return False


def close_all_positions():
    try:
        api.close_all()
        log("Closed all positions")
    except Exception as e:
        log(f"Failed to close positions: {e}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    global api

    import sys
    import io
    sys.stderr = io.StringIO()

    if (
        not CTRADER_HOST
        or not CTRADER_ACCOUNT
        or not CTRADER_PASSWORD
        or not CTRADER_BROKER
    ):
        print("=" * 60)
        print("ERROR: Missing cTrader FIX API credentials in .env")
        print("=" * 60)
        print("Required variables:")
        print("  CTRADER_HOST=host.ip")
        print("  CTRADER_ACCOUNT=your_login")
        print("  CTRADER_PASSWORD=your_password")
        print("  CTRADER_BROKER=broker_id")
        print("  CTRADER_PORT=5212 (default)")
        sys.exit(1)

    log("Initializing MFT Bot...")
    log(f"Host: {CTRADER_HOST}")
    log(f"Account: {CTRADER_ACCOUNT}")
    log(f"Broker: {CTRADER_BROKER}")

    try:
        api = Ctrader(CTRADER_HOST, CTRADER_ACCOUNT, CTRADER_PASSWORD)
        time.sleep(3)
    except Exception as e:
        print(f"=" * 60)
        print(f"ERROR: Failed to connect to cTrader: {e}")
        print("=" * 60)
        sys.exit(1)

    if not api.isconnected():
        print("=" * 60)
        print("ERROR: Not connected to cTrader FIX API")
        print("=" * 60)
        sys.exit(1)

    log("Connected to cTrader!")

    try:
        log("Subscribing to symbols...")
        for symbol in SYMBOL_MAP.values():
            api.subscribe(symbol)
        time.sleep(2)
    except Exception as e:
        log(f"Subscribe warning: {e}")

    try:
        positions = api.positions()
        log(f"Open positions: {len(positions)}")
        balance = DEFAULT_BALANCE
    except Exception as e:
        log(f"Failed to get positions: {e}")
        balance = DEFAULT_BALANCE

    log(f"Account balance: ${balance:,.2f} (demo)")

    models = load_all_models()

    if not models:
        print("No models loaded! Run train.py first.")
        sys.exit(1)

    log(f"Loaded {len(models)} models: {', '.join(models.keys())}")
    log(
        f"Thresholds - Opp: {OPP_THRESHOLD}, Dir: {DIR_LONG_THRESHOLD}/{DIR_SHORT_THRESHOLD}, Combined: {COMBINED_CONF_THRESHOLD}"
    )

    log("Bot started! Waiting for signals...")

    log("Fetching live prices...")
    try:
        prices = get_latest_prices()
        for ctrader_symbol, price_data in prices.items():
            bid = price_data.get("bid", 0)
            ask = price_data.get("ask", 0)
            if bid and ask:
                update_tick_buffer(ctrader_symbol.lower(), bid, ask)
        log(f"Updated tick buffers: {len(tick_buffers)} symbols")
    except Exception as e:
        log(f"Failed to fetch prices: {e}")

    try:
        while True:
            try:
                prices = get_latest_prices()
                for ctrader_symbol, price_data in prices.items():
                    bid = price_data.get("bid", 0)
                    ask = price_data.get("ask", 0)
                    if bid and ask:
                        update_tick_buffer(ctrader_symbol.lower(), bid, ask)

                signals = []

                min_ticks = min(len(tick_buffers[s]) for s in tick_buffers)
                if min_ticks < 50:
                    log(
                        f"Tick buffers: {', '.join(f'{s}:{len(tick_buffers[s])}' for s in tick_buffers)}"
                    )

                for symbol, model_data in models.items():
                    features = compute_live_features(symbol)
                    if features is None:
                        continue

                    scaler = model_data["scaler"]
                    feature_names = model_data["features"]
                    model_opp = model_data["model_opp"]
                    model_dir = model_data["model_dir"]

                    X = np.array([[features.get(f, 0) for f in feature_names]])
                    X = scaler.transform(X)

                    opp_pred = model_opp.predict_proba(X)[0, 1]
                    dir_pred = model_dir.predict_proba(X)[0, 1]

                    combined_conf = calculate_combined_confidence(opp_pred, dir_pred)

                    meets_thresh = "YES" if (opp_pred > OPP_THRESHOLD and 
                                            (dir_pred > DIR_LONG_THRESHOLD or dir_pred < DIR_SHORT_THRESHOLD) and
                                            combined_conf > COMBINED_CONF_THRESHOLD) else "NO"
                    log(f"{symbol.upper()}: opp={opp_pred:.2f}, dir={dir_pred:.2f}, combined={combined_conf:.2f} -> {meets_thresh}")

                    if opp_pred > OPP_THRESHOLD:
                        if dir_pred > DIR_LONG_THRESHOLD:
                            signal_type = "LONG"
                            signals.append(
                                (symbol, combined_conf, signal_type, dir_pred)
                            )
                        elif dir_pred < DIR_SHORT_THRESHOLD:
                            signal_type = "SHORT"
                            signals.append(
                                (symbol, combined_conf, signal_type, dir_pred)
                            )

                trade_signals = [s for s in signals if s[1] > COMBINED_CONF_THRESHOLD]
                trade_signals.sort(key=lambda x: x[1], reverse=True)

                if trade_signals:
                    log(f"Found {len(trade_signals)} trade signal(s):")
                    for symbol, conf, signal_type, direction in trade_signals:
                        if has_open_position(symbol):
                            log(f"  {symbol.upper()}: Position already open, skipping")
                            continue
                        
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
        log("Shutdown complete")


if __name__ == "__main__":
    main()
