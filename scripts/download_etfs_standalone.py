"""Standalone ETF data downloader — can run on any machine with yfinance.

Usage:
    pip install yfinance pandas pyarrow
    python download_etfs_standalone.py

Output: saves parquet files to ./etf_data/ directory.
Copy the parquet files to the main project's data/cache/ directory.
"""
import yfinance as yf
import pandas as pd
import os
import time

OUTPUT_DIR = "./etf_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = {
    "XLK": "XLK",   # Technology
    "XLF": "XLF",   # Financials
    "XLE": "XLE",   # Energy
    "XLV": "XLV",   # Healthcare
    "XLI": "XLI",   # Industrials
    "XLC": "XLC",   # Communication Services
    "XLY": "XLY",   # Consumer Discretionary
    "XLP": "XLP",   # Consumer Staples
}

START_DATE = "2010-01-01"

for name, ticker in TICKERS.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    if os.path.exists(out_path):
        print(f"{name}: already exists, skipping")
        continue

    print(f"Downloading {name} ({ticker})...")
    try:
        df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
        if len(df) == 0:
            print(f"  WARNING: no data returned for {name}")
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows -> {out_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    time.sleep(2)

print("\nDone. Copy all .parquet files from ./etf_data/ to your project's data/cache/ directory.")
