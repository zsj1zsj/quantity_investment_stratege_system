import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

from config import (
    SYMBOLS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE,
    LGBM_PARAMS, MODEL_SAVE_DIR,
)
from data.store import load, has_cache
from features.technical import (
    add_technical_features, add_cross_market_features,
    get_feature_columns, get_all_feature_columns,
)
from features.label import add_label
from model.evaluate import evaluate_window


def _load_cross_market():
    """Load VIX and TNX data if cached."""
    vix_df = load("VIX") if has_cache("VIX") else None
    tnx_df = load("TNX") if has_cache("TNX") else None
    return vix_df, tnx_df


def prepare_data(symbol_key: str) -> pd.DataFrame:
    """Load raw data, add features (including cross-market if available) and labels."""
    df = load(symbol_key)
    df = add_technical_features(df)

    vix_df, tnx_df = _load_cross_market()
    if vix_df is not None and tnx_df is not None:
        df = add_cross_market_features(df, vix_df, tnx_df)

    df = add_label(df)
    df = df.dropna()
    return df


def _get_feature_cols() -> list[str]:
    """Return appropriate feature columns based on available data."""
    vix_df, tnx_df = _load_cross_market()
    if vix_df is not None and tnx_df is not None:
        return get_all_feature_columns()
    return get_feature_columns()


def sliding_window_split(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate (train, test) splits using sliding window by trading days."""
    splits = []
    n = len(df)
    start = 0

    while start + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_end = start + TRAIN_WINDOW
        test_end = train_end + TEST_WINDOW

        train = df.iloc[start:train_end]
        test = df.iloc[train_end:test_end]

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

        start += STEP_SIZE

    return splits


def train_symbol(symbol_key: str) -> dict:
    """Train LightGBM with sliding window for one symbol."""
    print(f"\n{'='*50}")
    print(f"Training model for {symbol_key}")
    print(f"{'='*50}")

    df = prepare_data(symbol_key)
    feature_cols = _get_feature_cols()
    splits = sliding_window_split(df)

    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Number of sliding windows: {len(splits)}")

    all_metrics = []
    best_model = None
    best_auc = 0.0

    for i, (train_df, test_df) in enumerate(splits):
        train_range = f"{train_df.index[0].date()} ~ {train_df.index[-1].date()}"
        test_range = f"{test_df.index[0].date()} ~ {test_df.index[-1].date()}"
        print(f"\n  Window {i+1}: train [{train_range}], test [{test_range}]")

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        metrics = evaluate_window(model, X_test, y_test, window_name=f"Window {i+1}")
        all_metrics.append(metrics)

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_model = model

    # Save the best model
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_DIR / f"{symbol_key}_lgbm.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n  Best model saved -> {model_path} (AUC: {best_auc:.4f})")

    # Train final model on all data for prediction
    X_all = df[feature_cols]
    y_all = df["label"]
    final_model = LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X_all, y_all)
    final_path = MODEL_SAVE_DIR / f"{symbol_key}_lgbm_latest.pkl"
    joblib.dump(final_model, final_path)
    print(f"  Latest model (full data) saved -> {final_path}")

    return {
        "symbol": symbol_key,
        "windows": len(splits),
        "metrics": all_metrics,
        "feature_importance": dict(
            zip(feature_cols, final_model.feature_importances_)
        ),
    }


def train_all() -> list[dict]:
    """Train models for all configured symbols."""
    results = []
    for key in SYMBOLS:
        result = train_symbol(key)
        results.append(result)
    return results
