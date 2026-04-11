import time
import yfinance as yf
import pandas as pd

from config import SYMBOLS, CROSS_MARKET_SYMBOLS, DATA_START_DATE

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


def fetch_index_data(symbol_key: str, ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data for a given index from Yahoo Finance."""
    for attempt in range(1, MAX_RETRIES + 1):
        df = yf.download(ticker, start=DATA_START_DATE, auto_adjust=True, progress=False)
        if len(df) > 0:
            break
        if attempt < MAX_RETRIES:
            print(f"  Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if len(df) == 0:
        raise RuntimeError(f"Failed to fetch data for {symbol_key} ({ticker}) after {MAX_RETRIES} attempts")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def fetch_all() -> dict[str, pd.DataFrame]:
    """Fetch data for all configured symbols including cross-market."""
    result = {}

    # Main symbols
    for key, ticker in SYMBOLS.items():
        print(f"Fetching {key} ({ticker})...")
        result[key] = fetch_index_data(key, ticker)
        print(f"  Got {len(result[key])} rows")
        time.sleep(2)

    # Cross-market symbols
    for key, ticker in CROSS_MARKET_SYMBOLS.items():
        print(f"Fetching {key} ({ticker})...")
        result[key] = fetch_index_data(key, ticker)
        print(f"  Got {len(result[key])} rows")
        time.sleep(2)

    return result
