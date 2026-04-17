import gc
import glob
import json
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from collections import deque

import joblib
import numpy as np
import pandas as pd
import psutil
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
    "nzusd": "NZDUSD",
    "eurjpy": "EURJPY",
    "gbpjpy": "GBPJPY",
    "audjpy": "AUDJPY",
    "eurgbp": "EURGBP",
}

TICK_HISTORY = 50


def get_active_symbols(performance_file="saved_models/performance.json", top_n=6):
    """
    Load top performing symbols from backtest results.
    Sorts by profit_factor and returns top N symbols.
    Falls back to all symbols if file not found.
    """
    try:
        with open(performance_file, "r") as f:
            results = json.load(f)

        sorted_symbols = sorted(
            results.items(), key=lambda x: x[1].get("profit_factor", 0), reverse=True
        )

        top_dict = {}
        for symbol, _ in sorted_symbols[:top_n]:
            top_dict[symbol] = SYMBOL_MAP.get(symbol, symbol.upper())

        return top_dict
    except Exception as e:
        return SYMBOL_MAP


ACTIVE_SYMBOLS = get_active_symbols(top_n=6)

tick_buffers = {symbol: deque(maxlen=TICK_HISTORY) for symbol in ACTIVE_SYMBOLS}


def log_memory_usage():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    log(f"Memory: {mem_mb:.1f}MB")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("API call timed out")


def safe_api_call(func, *args, timeout=5, **kwargs):
    original_timeout = signal.alarm(0)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutException:
        signal.alarm(0)
        return None
    except Exception as e:
        signal.alarm(0)
        return None


def check_connection():
    try:
        return api.isconnected()
    except:
        return False


def reconnect(max_retries=5):
    global api
    for attempt in range(max_retries):
        wait_time = (2**attempt) + random.uniform(0, 1)
        log(
            f"Reconnecting... attempt {attempt + 1}/{max_retries}, wait {wait_time:.1f}s"
        )
        time.sleep(wait_time)
        try:
            api = Ctrader(CTRADER_HOST, CTRADER_ACCOUNT, CTRADER_PASSWORD)
            time.sleep(2)
            if api.isconnected():
                log("Reconnected successfully")
                for symbol in ACTIVE_SYMBOLS.values():
                    safe_api_call(api.subscribe, symbol)
                time.sleep(1)
                return True
        except:
            continue
    log("Max retries reached")
    return False


OPP_THRESHOLD = 0.70
DIR_LONG_THRESHOLD = 0.55
DIR_SHORT_THRESHOLD = 0.45
COMBINED_CONF_THRESHOLD = 0.35

RISK_PER_TRADE = 0.02
SL_PIPS = 10
TP_PIPS = 20

FIXED_LOT_SIZE = 100.0  # Fixed lot size for all trades

# Cash Buffer
CASH_BUFFER_PERCENT = 0.20  # Reserve 20% as unused

# Dynamic Position Sizing
BASE_LOT_SIZE = 100.0
MIN_EQUITY_THRESHOLD = 10000

# Max Drawdown
MAX_DRAWDOWN_PERCENT = 0.05  # 5% loss triggers pause

# Max Concurrent Positions
MAX_CONCURRENT_POSITIONS = 4

# Margin Utilization
MAX_MARGIN_USAGE_PERCENT = 0.50

# Correlation Filter
CORRELATED_PAIRS = {
    "EURUSD": ["GBPUSD", "USDCHF", "EURGBP"],
    "GBPUSD": ["EURUSD", "USDCHF", "EURGBP"],
    "USDJPY": ["EURJPY", "GBPJPY", "AUDJPY"],
    "AUDUSD": ["NZDUSD"],
    "USDCAD": [],
    "USDCHF": ["EURUSD", "GBPUSD"],
    "NZDUSD": ["AUDUSD"],
    "EURJPY": ["GBPJPY", "USDJPY"],
    "GBPJPY": ["EURJPY", "USDJPY"],
    "AUDJPY": ["USDJPY"],
    "EURGBP": ["EURUSD", "GBPUSD"],
}

# State tracking
initial_equity = None
peak_equity = None
drawdown_paused = False

MARGIN_CALL_ACTIVE = False
MARGIN_RECOVERY_WAIT = 60
MARGIN_MAX_WAIT = 600
MARGIN_CHECK_COUNT = 0

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


def detect_margin_call():
    try:
        positions = safe_api_call(api.positions, timeout=3)

        # Only consider it a margin call if we get actual position data with state field
        if positions is not None and len(positions) > 0:
            for pos in positions:
                state = pos.get("state", "")
                if state in ["margin_call", "liquidating", "stop_out"]:
                    return True

        # If positions returns empty list (normal state), that's not a margin call
        # Only return True if we get explicit margin call state
        return False

    except:
        return False  # Don't treat API errors as margin call


def get_account_equity():
    try:
        account_info = safe_api_call(api.accountInfo)
        if account_info and "balance" in account_info:
            return float(account_info["balance"])
        return None
    except:
        return None


def calculate_max_lot_size(equity):
    if equity is None:
        equity = 1000000
    available_for_trading = equity * (1 - CASH_BUFFER_PERCENT)
    risk_amount = available_for_trading * RISK_PER_TRADE
    lot = risk_amount / (SL_PIPS * 10)
    max_lot = min(lot, BASE_LOT_SIZE)
    return max(max_lot, 0.01)


def calculate_confidence_lot(equity, initial, confidence):
    base_lot = calculate_max_lot_size(equity)
    dynamic_lot = calculate_dynamic_lot_size(equity, initial)
    base = min(base_lot, dynamic_lot)
    # Scale lot by confidence (0.3 to 1.0 multiplier)
    conf_multiplier = max(min(confidence * 2, 1.0), 0.3)
    return base * conf_multiplier


def calculate_dynamic_lot_size(equity, initial):
    if equity is None or initial is None or initial == 0:
        return BASE_LOT_SIZE
    ratio = equity / initial
    min_ratio = 0.2  # Minimum 20% of max lot
    adjusted_ratio = max(ratio, min_ratio)
    return BASE_LOT_SIZE * adjusted_ratio


def check_max_drawdown(equity, peak, paused):
    if equity is None:
        return True, peak, paused

    if peak is None:
        peak = equity

    if equity > peak:
        peak = equity
        if paused:
            paused = False
        return True, peak, False

    drawdown = (peak - equity) / peak
    if drawdown >= MAX_DRAWDOWN_PERCENT and not paused:
        return False, peak, True

    return True, peak, paused


def get_margin_usage():
    try:
        positions = safe_api_call(api.positions, timeout=3)
        if not positions:
            return 0.0

        total_margin = 0
        for pos in positions:
            margin = pos.get("margin", 0)
            if margin:
                total_margin += float(margin)

        account_info = safe_api_call(api.accountInfo)
        if account_info and "balance" in account_info:
            equity = float(account_info["balance"])
            if equity > 0:
                return total_margin / equity

        return 0.0
    except:
        return 0.0


def get_open_positions_count():
    try:
        positions = safe_api_call(api.positions, timeout=3)
        return len(positions) if positions else 0
    except:
        return 0


def has_correlated_position(symbol, direction):
    try:
        positions = safe_api_call(api.positions, timeout=3)
        if not positions:
            return False

        correlated = CORRELATED_PAIRS.get(symbol, [])
        direction_map = {"LONG": "buy", "SHORT": "sell"}
        trade_direction = direction_map.get(direction, "")

        for pos in positions:
            pos_symbol = pos.get("symbol", "").upper()
            if pos_symbol in correlated:
                pos_type = pos.get("type", "").lower()
                if pos_type == trade_direction:
                    return True

        return False
    except:
        return False


def load_all_models():
    models = {}
    active = set(ACTIVE_SYMBOLS.keys())
    model_files = glob.glob("saved_models/*_xgb_model.pkl")

    for f in model_files:
        symbol = os.path.basename(f).replace("_xgb_model.pkl", "")
        if symbol not in active:
            continue
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
    for symbol in ACTIVE_SYMBOLS.values():
        try:
            q = safe_api_call(api.quote, symbol)
            if q:
                if "bid" in q and "ask" in q:
                    prices[symbol] = q
                elif "last" in q:
                    prices[symbol] = {"bid": q["last"], "ask": q["last"]}
        except:
            pass

    if prices:
        return prices

    try:
        positions = safe_api_call(api.positions)
        if positions:
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
    # Find the correct key in tick_buffers (could be 'nzusd' vs 'nzdusd')
    buffer_key = None
    for key, val in ACTIVE_SYMBOLS.items():
        if val.upper() == symbol.upper():
            buffer_key = key
            break

    if buffer_key is None:
        buffer_key = symbol.lower()

    if buffer_key not in tick_buffers:
        return

    mid = (bid + ask) / 2
    tick_buffers[buffer_key].append(
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

    # Micro-structure features for HFT
    tick_dir = np.sign(df["close"].diff())
    tick_dir = tick_dir.replace(0, np.nan).ffill().fillna(0)
    features["tick_direction_imbalance"] = tick_dir.rolling(window=20, min_periods=1).mean().iloc[-1]

    spread_ma = df["tick_spread"].rolling(window=20, min_periods=1).mean().iloc[-1]
    features["spread_compression"] = 1 if df["tick_spread"].iloc[-1] < spread_ma * 0.5 else 0

    features["tick_acceleration"] = df["close"].diff().diff().iloc[-1]

    vwap = (df["close"] * df["tick_volume"]).rolling(window=50, min_periods=1).sum() / (
        df["tick_volume"].rolling(window=50, min_periods=1).sum() + 1e-9
    )
    features["vwap_deviation"] = (df["close"].iloc[-1] - vwap.iloc[-1]) / (features["tick_std"] + 1e-9)

    features["volume_weighted_imbalance"] = 0

    features["consecutive_bid"] = sum(1 for i in range(-5, 0) if df["close"].diff().iloc[i] > 0)
    features["consecutive_ask"] = sum(1 for i in range(-5, 0) if df["close"].diff().iloc[i] < 0)

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
    ctrader_symbol = ACTIVE_SYMBOLS.get(symbol, symbol.upper())

    try:
        positions = safe_api_call(api.positions)
        if positions:
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
            result = safe_api_call(api.buy, ctrader_symbol, lot, sl, tp)
            log(
                f"Trade BUY {ctrader_symbol} | Lot: {lot} | SL: {sl} | TP: {tp} | Result: {result}"
            )
        else:
            sl = round(ask + (sl_pips * pip_val), digits)
            tp = round(ask - (tp_pips * pip_val), digits)
            result = safe_api_call(api.sell, ctrader_symbol, lot, sl, tp)
            log(
                f"Trade SELL {ctrader_symbol} | Lot: {lot} | SL: {sl} | TP: {tp} | Result: {result}"
            )

        return True

    except Exception as e:
        log(f"Trade failed: {e}")
        return False


def has_open_position(symbol):
    ctrader_symbol = ACTIVE_SYMBOLS.get(symbol, symbol.upper())
    try:
        positions = safe_api_call(api.positions)
        if not positions:
            return False
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

    import io
    import sys

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

    global \
        MARGIN_CALL_ACTIVE, \
        MARGIN_CHECK_COUNT, \
        peak_equity, \
        drawdown_paused, \
        initial_equity

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
        for symbol in ACTIVE_SYMBOLS.values():
            safe_api_call(api.subscribe, symbol)
        time.sleep(2)
    except Exception as e:
        log(f"Subscribe warning: {e}")

    try:
        positions = safe_api_call(api.positions)
        log(f"Open positions: {len(positions) if positions else 0}")

        global initial_equity, peak_equity
        equity = get_account_equity()
        if equity:
            initial_equity = equity
            peak_equity = equity
            log(f"Initial equity: ${equity:,.2f}")
        else:
            initial_equity = 1000000
            peak_equity = 1000000
            log("Could not get equity, using default: $1,000,000")
    except Exception as e:
        log(f"Failed to get positions: {e}")

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
        cycle_count = 0
        while True:
            cycle_count += 1
            if cycle_count % 10 == 0:
                gc.collect()
                log_memory_usage()

            try:
                if not check_connection():
                    log("Connection lost, attempting reconnect...")
                    if not reconnect():
                        log("Reconnection failed, exiting...")
                        break

                if not MARGIN_CALL_ACTIVE and detect_margin_call():
                    log("MARGIN CALL DETECTED! Closing positions...")
                    MARGIN_CALL_ACTIVE = True
                    MARGIN_CHECK_COUNT = 0

                    try:
                        safe_api_call(api.close_all)
                        log("All positions closed")
                    except Exception as e:
                        log(f"Close failed: {e}")

                    log("Waiting for margin recovery...")

                if MARGIN_CALL_ACTIVE:
                    if detect_margin_call():
                        MARGIN_CHECK_COUNT += 1
                        log(
                            f"Margin call active - waiting ({MARGIN_CHECK_COUNT}/10)..."
                        )

                        if MARGIN_CHECK_COUNT >= 10:
                            log("Max wait exceeded, exiting...")
                            break

                        time.sleep(MARGIN_RECOVERY_WAIT)
                        continue
                    else:
                        log("Margin recovered! Resuming trading...")
                        MARGIN_CALL_ACTIVE = False
                        MARGIN_CHECK_COUNT = 0
                        for symbol in ACTIVE_SYMBOLS:
                            tick_buffers[symbol].clear()
                        continue

                # Get current equity
                equity = get_account_equity()

                # Check drawdown
                can_trade, peak_equity, drawdown_paused = check_max_drawdown(
                    equity, peak_equity, drawdown_paused
                )
                if not can_trade:
                    log(
                        f"Max drawdown ({MAX_DRAWDOWN_PERCENT * 100}%) reached - pausing for today"
                    )
                    try:
                        safe_api_call(api.close_all)
                    except:
                        pass
                    break

                # Check equity minimum
                if equity and equity < MIN_EQUITY_THRESHOLD:
                    log(
                        f"Equity ${equity:,.2f} below minimum ${MIN_EQUITY_THRESHOLD:,} - stopping..."
                    )
                    break

                # Check margin usage
                margin_usage = get_margin_usage()
                if margin_usage > MAX_MARGIN_USAGE_PERCENT:
                    log(
                        f"Margin usage {margin_usage * 100:.1f}% > {MAX_MARGIN_USAGE_PERCENT * 100:.0f}%, skipping..."
                    )

                prices = get_latest_prices()

                if not prices:
                    log(f"No prices received, will retry...")
                    time.sleep(3)
                    continue

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

                    meets_thresh = (
                        "YES"
                        if (
                            opp_pred > OPP_THRESHOLD
                            and (
                                dir_pred > DIR_LONG_THRESHOLD
                                or dir_pred < DIR_SHORT_THRESHOLD
                            )
                            and combined_conf > COMBINED_CONF_THRESHOLD
                        )
                        else "NO"
                    )
                    log(
                        f"{symbol.upper()}: opp={opp_pred:.2f}, dir={dir_pred:.2f}, combined={combined_conf:.2f} -> {meets_thresh}"
                    )

                    if meets_thresh == "YES":
                        # Determine signal direction first for correlation check
                        if dir_pred > DIR_LONG_THRESHOLD:
                            current_signal_type = "LONG"
                        elif dir_pred < DIR_SHORT_THRESHOLD:
                            current_signal_type = "SHORT"
                        else:
                            current_signal_type = None

                        # Check correlation with current direction
                        if current_signal_type and has_correlated_position(
                            symbol, current_signal_type
                        ):
                            log(f"Correlated position exists for {symbol}, skipping")

                        # Check max concurrent positions
                        elif get_open_positions_count() >= MAX_CONCURRENT_POSITIONS:
                            log(
                                f"Max positions ({MAX_CONCURRENT_POSITIONS}) reached for {symbol}, skipping"
                            )

                        # Add to signals if passing all checks
                        elif opp_pred > OPP_THRESHOLD:
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

                        equity = get_account_equity()
                        lot = calculate_confidence_lot(equity, initial_equity, conf)

                        if signal_type == "LONG":
                            place_trade("BUY", symbol, lot, SL_PIPS, TP_PIPS)
                        else:
                            place_trade("SELL", symbol, lot, SL_PIPS, TP_PIPS)

                        log(
                            f"  {symbol.upper()}: {signal_type} | Conf: {conf:.3f} | Lot: {lot:.2f}"
                        )

                else:
                    time.sleep(3)

            except Exception as e:
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                log(f"Error: {e} | File: {tb.filename} | Line: {tb.lineno}")
                time.sleep(3)

    except KeyboardInterrupt:
        log("Bot stopped by user")

    finally:
        log("Closing all positions...")
        close_all_positions()
        log("Shutdown complete")


if __name__ == "__main__":
    main()
