import json
import joblib
import pandas as pd

from config import SYMBOLS, MODEL_SAVE_DIR, PROB_BUY_THRESHOLD, PROB_SELL_THRESHOLD
from data.store import load, has_cache
from features.technical import (
    add_technical_features, add_cross_market_features,
    get_feature_columns, get_all_feature_columns,
)
from strategy.spec import MarketContext
from strategy.engine import evaluate


def predict_symbol(symbol_key: str) -> dict:
    """Generate prediction for a single symbol using latest model + strategy layer."""
    model_path = MODEL_SAVE_DIR / f"{symbol_key}_lgbm_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model for {symbol_key}. Run 'train' first."
        )

    model = joblib.load(model_path)

    df = load(symbol_key)
    df = add_technical_features(df)

    vix_df = load("VIX") if has_cache("VIX") else None
    tnx_df = load("TNX") if has_cache("TNX") else None
    if vix_df is not None and tnx_df is not None:
        df = add_cross_market_features(df, vix_df, tnx_df)
        feature_cols = get_all_feature_columns()
    else:
        feature_cols = get_feature_columns()

    df = df.dropna()
    latest = df.iloc[[-1]]
    X = latest[feature_cols]

    prob_up = float(model.predict_proba(X)[0][1])

    row = latest.iloc[0]

    # Build market context for strategy evaluation
    ctx = MarketContext(
        date=str(latest.index[0].date()),
        close=row["close"],
        prob_up=prob_up,
        ma5=row["ma5"],
        ma20=row["ma20"],
        close_ma20_ratio=row["close_ma20_ratio"],
        rsi=row["rsi"],
        volatility_5d=row["volatility_5d"],
        volatility_20d=row["volatility_20d"],
        vol_ratio=row["vol_ratio"],
        macd_hist=row["macd_hist"],
        vix=row.get("vix", 0),
    )

    decision = evaluate(ctx)

    # Determine signal strength
    if prob_up > 0.8:
        signal_strength = "very_high"
    elif prob_up > PROB_BUY_THRESHOLD:
        signal_strength = "high"
    elif prob_up > 0.5:
        signal_strength = "medium"
    elif prob_up > PROB_SELL_THRESHOLD:
        signal_strength = "low"
    else:
        signal_strength = "very_low"

    # Volatility status for risk note
    vol_ratio = row["vol_ratio"]
    if vol_ratio > 1.5:
        vol_status = "high"
    elif vol_ratio < 0.5:
        vol_status = "low"
    else:
        vol_status = "normal"

    return {
        "date": str(latest.index[0].date()),
        "symbol": symbol_key,
        "signal_strength": signal_strength,
        "probability": round(prob_up, 4),
        "suggestion": decision.action.value,
        "position_size": decision.position_size,
        "regime": decision.regime,
        "risk_note": f"vol_ratio={vol_ratio:.2f}, volatility {vol_status}",
        "model_version": "v2.0-lgbm",
        "details": {
            "close": round(row["close"], 2),
            "ticker": SYMBOLS[symbol_key],
            "ma_trend": "bullish" if row["close_ma20_ratio"] > 1 else "bearish",
            "rsi": round(row["rsi"], 1),
            "macd_hist": round(row["macd_hist"], 4),
            "decision_reason": decision.reason,
        },
    }


def predict_all() -> list[dict]:
    """Generate predictions for all configured symbols."""
    results = []
    for key in SYMBOLS:
        result = predict_symbol(key)
        results.append(result)
    return results
