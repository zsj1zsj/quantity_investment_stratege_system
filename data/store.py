import pandas as pd

from config import DATA_CACHE_DIR


def save(symbol_key: str, df: pd.DataFrame) -> None:
    """Save dataframe to parquet cache."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_CACHE_DIR / f"{symbol_key}.parquet"
    df.to_parquet(path)
    print(f"Saved {symbol_key} -> {path}")


def load(symbol_key: str) -> pd.DataFrame:
    """Load dataframe from parquet cache."""
    path = DATA_CACHE_DIR / f"{symbol_key}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {symbol_key}. Run 'fetch' first.")
    return pd.read_parquet(path)


def has_cache(symbol_key: str) -> bool:
    """Check if cached data exists."""
    return (DATA_CACHE_DIR / f"{symbol_key}.parquet").exists()
