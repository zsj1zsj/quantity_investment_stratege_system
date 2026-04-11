"""Signal quality validation: bucket analysis of model predictions vs actual returns.

Verifies that high-confidence predictions actually produce positive expected returns.
This is a Gate 1 prerequisite before deploying the strategy.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE
from model.train import _get_feature_cols


PROB_BINS = [0.0, 0.3, 0.5, 0.7, 1.0]
BIN_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">0.7"]


def validate_signals(df: pd.DataFrame, symbol: str) -> dict:
    """Run walk-forward signal validation with bucket analysis.

    For each sliding window, train on train set and predict on test set.
    Collect all out-of-sample predictions, then analyze by probability bucket.
    """
    feature_cols = _get_feature_cols()
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

    # Remove NaN returns
    valid_mask = ~np.isnan(all_returns)
    all_probs = all_probs[valid_mask]
    all_returns = all_returns[valid_mask]

    # Bucket analysis
    bucket_results = {}
    bin_indices = np.digitize(all_probs, PROB_BINS) - 1
    bin_indices = np.clip(bin_indices, 0, len(BIN_LABELS) - 1)

    for i, label in enumerate(BIN_LABELS):
        mask = bin_indices == i
        if mask.sum() == 0:
            bucket_results[label] = {
                "count": 0,
                "mean_return": 0.0,
                "median_return": 0.0,
                "positive_pct": 0.0,
                "std_return": 0.0,
            }
            continue

        bucket_returns = all_returns[mask]
        bucket_results[label] = {
            "count": int(mask.sum()),
            "mean_return": float(np.mean(bucket_returns)),
            "median_return": float(np.median(bucket_returns)),
            "positive_pct": float(np.mean(bucket_returns > 0)),
            "std_return": float(np.std(bucket_returns)),
        }

    # Gate 1 check: prob>0.7 bucket mean return > 0.5%
    high_conf = bucket_results.get(">0.7", {})
    gate1_pass = high_conf.get("mean_return", 0) > 0.005

    return {
        "symbol": symbol,
        "total_samples": len(all_probs),
        "buckets": bucket_results,
        "gate1_signal_quality": gate1_pass,
    }


def print_signal_validation(result: dict) -> None:
    """Print formatted signal validation report."""
    print(f"\n{'='*60}")
    print(f"  Signal Quality Validation: {result['symbol']}")
    print(f"{'='*60}")
    print(f"  Total out-of-sample predictions: {result['total_samples']}")
    print()

    header = f"  {'Bucket':>10s} | {'Count':>6s} | {'Mean Ret':>9s} | {'Median':>8s} | {'Win%':>6s} | {'Std':>8s}"
    print(header)
    print(f"  {'-'*56}")

    for label in BIN_LABELS:
        b = result["buckets"][label]
        print(f"  {label:>10s} | {b['count']:>6d} | {b['mean_return']:>8.3%} | "
              f"{b['median_return']:>7.3%} | {b['positive_pct']:>5.1%} | {b['std_return']:>7.3%}")

    print()
    gate = result["gate1_signal_quality"]
    high_conf = result["buckets"].get(">0.7", {})
    print(f"  Gate 1 Check: prob>0.7 mean return > 0.5%")
    print(f"    Result: {'PASS' if gate else 'FAIL'} (mean={high_conf.get('mean_return', 0):.3%})")
    print(f"{'='*60}")
