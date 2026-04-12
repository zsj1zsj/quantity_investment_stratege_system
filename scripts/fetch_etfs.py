"""Fetch ETF data with aggressive retry for rate limiting."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_index_data
from data.store import save, has_cache
from config import ETF_SYMBOLS

DELAY_BETWEEN = 30  # seconds between requests
MAX_RETRIES = 10
RETRY_DELAY = 60  # seconds on rate limit

for key, ticker in ETF_SYMBOLS.items():
    if has_cache(key):
        print(f"{key}: already cached, skipping")
        continue

    print(f"\nFetching {key} ({ticker})...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            import yfinance as yf
            df = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)
            if len(df) > 0:
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index.name = "date"
                save(key, df)
                print(f"  Got {len(df)} rows, saved.")
                break
            else:
                print(f"  Empty result, attempt {attempt}/{MAX_RETRIES}")
        except Exception as e:
            print(f"  Error: {e}, attempt {attempt}/{MAX_RETRIES}")

        wait = RETRY_DELAY * attempt
        print(f"  Waiting {wait}s before retry...")
        time.sleep(wait)
    else:
        print(f"  FAILED to fetch {key} after {MAX_RETRIES} attempts")

    time.sleep(DELAY_BETWEEN)

print("\nDone.")
