from datetime import datetime, timedelta
from tqdm import tqdm
import dukascopy_python as dk
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_NZD_USD,
    INSTRUMENT_FX_CROSSES_EUR_JPY,
    INSTRUMENT_FX_CROSSES_GBP_JPY,
    INSTRUMENT_FX_CROSSES_AUD_JPY,
    INSTRUMENT_FX_CROSSES_EUR_GBP,
)
import pandas as pd
import time
import os

SYMBOLS = [
    (INSTRUMENT_FX_MAJORS_EUR_USD, "eurusd"),
    (INSTRUMENT_FX_MAJORS_GBP_USD, "gbpusd"),
    (INSTRUMENT_FX_MAJORS_USD_JPY, "usdjpy"),
    (INSTRUMENT_FX_MAJORS_AUD_USD, "audusd"),
    (INSTRUMENT_FX_MAJORS_USD_CAD, "usdcad"),
    (INSTRUMENT_FX_MAJORS_USD_CHF, "usdchf"),
    (INSTRUMENT_FX_MAJORS_NZD_USD, "nzusd"),
    (INSTRUMENT_FX_CROSSES_EUR_JPY, "eurjpy"),
    (INSTRUMENT_FX_CROSSES_GBP_JPY, "gbpjpy"),
    (INSTRUMENT_FX_CROSSES_AUD_JPY, "audjpy"),
    (INSTRUMENT_FX_CROSSES_EUR_GBP, "eurgbp"),
]

END = datetime.now()
START = END - timedelta(days=90)

INTERVAL = dk.INTERVAL_TICK
OFFER_SIDE = dk.OFFER_SIDE_BID

os.makedirs("data/raw_ticks", exist_ok=True)

def download_symbol(instrument, symbol, start, end):
    output_path = f"data/raw_ticks/{symbol}.csv"
    dates = [start + timedelta(days=i) for i in range((end - start).days)]
    
    all_ticks = []
    for day_start in tqdm(dates, desc=symbol, unit="day"):
        day_end = day_start + timedelta(days=1)
        try:
            df = dk.fetch(
                instrument=instrument,
                interval=INTERVAL,
                offer_side=OFFER_SIDE,
                start=day_start,
                end=day_end,
            )
            if df is not None and len(df) > 0:
                all_ticks.append(df)
        except:
            pass
        time.sleep(0.2)
    
    if all_ticks:
        combined = pd.concat(all_ticks, ignore_index=False)
        combined = combined.reset_index()
        combined.columns = ["timestamp", "bidPrice", "askPrice", "bidVolume", "askVolume"]
        combined.to_csv(output_path, sep="\t", index=False)
        return len(combined)
    return 0

if __name__ == "__main__":
    print(f"Downloading {INTERVAL} data: {START.date()} to {END.date()}")
    print(f"{len(SYMBOLS)} pairs, {(END-START).days} days\n")
    
    for idx, (instrument, symbol) in enumerate(SYMBOLS):
        print(f"[{idx+1}/{len(SYMBOLS)}] {symbol.upper()}...", end=" ")
        count = download_symbol(instrument, symbol, START, END)
        print(f"✓ {count:,} ticks")
        if idx < len(SYMBOLS) - 1:
            time.sleep(1)
    
    print("\n✅ All complete!")