import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

from config import (
    MA_WINDOWS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    VOLATILITY_WINDOWS,
    VOLUME_AVG_WINDOW,
    RETURN_WINDOWS,
)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-layer technical features to OHLCV dataframe.

    Layer 1 - Trend: MA5, MA20, close/MA20, MA5/MA20
    Layer 2 - Momentum: return_5d, return_10d, momentum_accel
    Layer 3 - Volatility: volatility_5d, volatility_20d, vol_ratio
    Layer 4 - Classic: RSI(14), MACD(12,26,9)
    """
    df = df.copy()
    close = df["close"]
    volume = df["volume"]
    daily_return = close.pct_change()

    # --- Layer 1: Trend ---
    for w in MA_WINDOWS:
        sma = SMAIndicator(close, window=w)
        df[f"ma{w}"] = sma.sma_indicator()
        df[f"close_ma{w}_ratio"] = close / df[f"ma{w}"]

    # MA5/MA20 cross signal (short-term vs long-term trend)
    if 5 in MA_WINDOWS and 20 in MA_WINDOWS:
        df["ma5_ma20_ratio"] = df["ma5"] / df["ma20"]

    # --- Layer 2: Momentum ---
    for w in RETURN_WINDOWS:
        df[f"return_{w}d"] = close.pct_change(periods=w)

    # Momentum acceleration
    if 5 in RETURN_WINDOWS and 10 in RETURN_WINDOWS:
        df["momentum_accel"] = df["return_5d"] - df["return_10d"]

    # --- Layer 3: Volatility ---
    for w in VOLATILITY_WINDOWS:
        df[f"volatility_{w}d"] = daily_return.rolling(window=w).std()

    # Volatility ratio (contraction/expansion)
    if 5 in VOLATILITY_WINDOWS and 20 in VOLATILITY_WINDOWS:
        df["vol_ratio"] = df["volatility_5d"] / df["volatility_20d"]

    # --- Layer 4: Classic indicators ---
    rsi = RSIIndicator(close, window=RSI_PERIOD)
    df["rsi"] = rsi.rsi()

    macd = MACD(close, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- Volume derived features ---
    vol_ma5 = volume.rolling(window=5).mean()
    vol_ma20 = volume.rolling(window=20).mean()

    # Volume ratio vs 5-day average (existing)
    df["volume_ratio"] = volume / vol_ma5

    # Volume 20-day Z-score
    vol_std20 = volume.rolling(window=20).std()
    df["volume_zscore_20d"] = (volume - vol_ma20) / vol_std20

    # Daily return
    df["return_1d"] = daily_return

    return df


def add_cross_market_features(df: pd.DataFrame, vix_df: pd.DataFrame, tnx_df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-market factors: VIX and Treasury yield features.

    Layer 5 - Cross-market:
      - vix: VIX absolute value
      - vix_change_5d: VIX 5-day change rate
      - tnx: 10-year Treasury yield
      - tnx_change_5d: yield 5-day change
      - vix_x_return_5d: VIX * return_5d interaction term
    """
    df = df.copy()

    # Align VIX data to df index (forward-fill for non-trading days)
    vix_close = vix_df["close"].reindex(df.index, method="ffill")
    df["vix"] = vix_close
    df["vix_change_5d"] = vix_close.pct_change(periods=5)

    # Align Treasury yield data
    tnx_close = tnx_df["close"].reindex(df.index, method="ffill")
    df["tnx"] = tnx_close
    df["tnx_change_5d"] = tnx_close.diff(periods=5)  # absolute change, not pct

    # Interaction term: VIX * 5-day return
    if "return_5d" in df.columns:
        df["vix_x_return_5d"] = df["vix"] * df["return_5d"]

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used by the model."""
    cols = []

    # Layer 1: Trend
    for w in MA_WINDOWS:
        cols.append(f"close_ma{w}_ratio")
    cols.append("ma5_ma20_ratio")

    # Layer 2: Momentum
    for w in RETURN_WINDOWS:
        cols.append(f"return_{w}d")
    cols.append("momentum_accel")

    # Layer 3: Volatility
    for w in VOLATILITY_WINDOWS:
        cols.append(f"volatility_{w}d")
    cols.append("vol_ratio")

    # Layer 4: Classic
    cols.append("rsi")
    cols.extend(["macd", "macd_signal", "macd_hist"])

    # Volume features
    cols.extend(["volume_ratio", "volume_zscore_20d"])

    # Other
    cols.append("return_1d")

    return cols


def get_all_feature_columns() -> list[str]:
    """Return feature columns including cross-market factors."""
    cols = get_feature_columns()
    cols.extend(["vix", "vix_change_5d", "tnx", "tnx_change_5d", "vix_x_return_5d"])
    return cols
