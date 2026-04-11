"""Experiment: test different label definitions and run Gate 1 validation.

Tests combinations of FORWARD_DAYS and LABEL_THRESHOLD to find
which produces the strongest signal differentiation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE
from data.store import load
from features.technical import add_technical_features, get_feature_columns


LABEL_CONFIGS = [
    # (forward_days, threshold, description)
    (5, 0.01, "5d >1% (current)"),
    (5, 0.015, "5d >1.5%"),
    (10, 0.02, "10d >2%"),
    (10, 0.025, "10d >2.5%"),
    (20, 0.03, "20d >3%"),
    (20, 0.04, "20d >4%"),
    (15, 0.02, "15d >2%"),
]

PROB_BINS = [0.0, 0.3, 0.5, 0.7, 1.0]
BIN_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">0.7"]


def make_label(df: pd.DataFrame, forward_days: int, threshold: float) -> pd.DataFrame:
    df = df.copy()
    future_return = df["close"].shift(-forward_days) / df["close"] - 1
    df["future_return"] = future_return
    df["label"] = (future_return > threshold).astype(float)
    df.loc[df["future_return"].isna(), "label"] = float("nan")
    return df


def run_signal_validation(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    n = len(df)
    all_probs = []
    all_returns = []

    start = 0
    while start + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_end = start + TRAIN_WINDOW
        test_end = min(train_end + TEST_WINDOW, n)

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        returns = test_df["future_return"].values

        all_probs.extend(probs)
        all_returns.extend(returns)

        start += STEP_SIZE

    all_probs = np.array(all_probs)
    all_returns = np.array(all_returns)

    valid = ~np.isnan(all_returns)
    all_probs = all_probs[valid]
    all_returns = all_returns[valid]

    bin_indices = np.digitize(all_probs, PROB_BINS) - 1
    bin_indices = np.clip(bin_indices, 0, len(BIN_LABELS) - 1)

    results = {}
    for i, label in enumerate(BIN_LABELS):
        mask = bin_indices == i
        if mask.sum() == 0:
            results[label] = {"count": 0, "mean": 0, "median": 0, "win_pct": 0}
            continue
        r = all_returns[mask]
        results[label] = {
            "count": int(mask.sum()),
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "win_pct": float(np.mean(r > 0)),
        }
    return results


def main():
    feature_cols = get_feature_columns()

    for symbol in ["SP500", "NASDAQ"]:
        print(f"\n{'='*70}")
        print(f"  {symbol} - Label Definition Search")
        print(f"{'='*70}")

        raw_df = load(symbol)
        raw_df = add_technical_features(raw_df)

        for fwd, thresh, desc in LABEL_CONFIGS:
            df = make_label(raw_df, fwd, thresh)
            df = df.dropna()

            pos_rate = df["label"].mean()
            buckets = run_signal_validation(df, feature_cols)

            high = buckets[">0.7"]
            low = buckets["<0.3"]
            spread = high["mean"] - low["mean"]

            gate1 = "PASS" if high["mean"] > 0.005 else "FAIL"

            print(f"\n  {desc} (pos_rate={pos_rate:.1%})")
            print(f"    {'Bucket':>10s} | {'Count':>6s} | {'Mean':>8s} | {'Median':>8s} | {'Win%':>6s}")
            print(f"    {'-'*48}")
            for bl in BIN_LABELS:
                b = buckets[bl]
                print(f"    {bl:>10s} | {b['count']:>6d} | {b['mean']:>7.3%} | {b['median']:>7.3%} | {b['win_pct']:>5.1%}")
            print(f"    High-Low spread: {spread:.3%}")
            print(f"    Gate 1 (>0.7 mean > 0.5%): {gate1}")


if __name__ == "__main__":
    main()
