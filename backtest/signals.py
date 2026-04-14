"""Shared walk-forward signal collection.

Single source of truth for out-of-sample LightGBM predictions used by
multi-asset backtest, stability analysis, and sector analysis.

Raw uncalibrated probabilities are used here to preserve backtest integrity.
Calibration and smoothing are applied only at prediction time (model/predict.py).
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

from config import LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE
from model.train import _get_feature_cols


def collect_asset_signals(df: pd.DataFrame, symbol: str) -> dict[str, dict]:
    """Walk-forward signal collection for a single asset.

    Returns a dict keyed by date string, each value containing:
        date, close, prob, vix, symbol
    All test-window predictions are strictly out-of-sample.
    """
    feature_cols = _get_feature_cols()
    n = len(df)
    all_data = []
    start = 0

    while start + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_end = start + TRAIN_WINDOW
        test_end = min(train_end + TEST_WINDOW, n)
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(train_df[feature_cols], train_df["label"])
        probs = model.predict_proba(test_df[feature_cols])[:, 1]

        for idx, row_idx in enumerate(test_df.index):
            row = test_df.loc[row_idx]
            all_data.append({
                "date": row_idx,
                "close": row["close"],
                "prob": probs[idx],
                "vix": row.get("vix", 0),
                "symbol": symbol,
            })
        start += STEP_SIZE

    # Deduplicate overlapping windows — keep first occurrence
    seen: set[str] = set()
    unique = []
    for d in all_data:
        k = str(d["date"])
        if k not in seen:
            seen.add(k)
            unique.append(d)
    unique.sort(key=lambda x: x["date"])
    return {str(d["date"]): d for d in unique}


def collect_all_signals(
    prepared_dfs: dict[str, pd.DataFrame],
) -> dict[str, dict[str, dict]]:
    """Walk-forward signal collection for all assets.

    Returns a dict mapping symbol -> {date_str -> signal_dict}.
    """
    return {
        symbol: collect_asset_signals(df, symbol)
        for symbol, df in prepared_dfs.items()
    }
