"""Date-range historical prediction report.

Generates walk-forward predictions for every trading day in a given date
range, showing what signal the model would have output on each day.

Coverage:
  - Dates within walk-forward range (~2012 to ~2 months ago):
      Each prediction is strictly out-of-sample (model only saw past data).
  - Recent dates not yet covered by walk-forward:
      Uses the latest trained model (_lgbm_latest.pkl). Not strictly
      out-of-sample — labelled as [latest-model] in output.
"""
from __future__ import annotations

import joblib
import pandas as pd

from config import SYMBOLS, ETF_SYMBOLS, PROB_BUY_THRESHOLD, MODEL_SAVE_DIR
from data.store import load, has_cache
from features.technical import (
    add_technical_features, add_cross_market_features,
    get_feature_columns, get_all_feature_columns,
)
from model.train import prepare_data, _get_feature_cols
from backtest.signals import collect_asset_signals
from strategy.regime import detect_regime, Regime


def _collect_recent_signals(symbol: str, known_dates: set[str], start_date: str, end_date: str) -> list[dict]:
    """Predict recent dates not covered by walk-forward using the latest model.

    Loads raw data (without dropping unlabelled tail rows) and applies the
    latest trained model. These predictions are NOT strictly out-of-sample.
    """
    model_path = MODEL_SAVE_DIR / f"{symbol}_lgbm_latest.pkl"
    if not model_path.exists():
        return []

    model = joblib.load(model_path)

    df = load(symbol)
    df = add_technical_features(df)

    vix_df = load("VIX") if has_cache("VIX") else None
    tnx_df = load("TNX") if has_cache("TNX") else None
    if vix_df is not None and tnx_df is not None:
        df = add_cross_market_features(df, vix_df, tnx_df)
        feature_cols = get_all_feature_columns()
    else:
        feature_cols = get_feature_columns()

    # Drop rows with missing feature values (keep rows with missing label)
    df = df.dropna(subset=feature_cols)

    results = []
    for row_idx, row in df.iterrows():
        date_str = str(row_idx)[:10]
        if date_str in known_dates:
            continue
        if date_str < start_date or date_str > end_date:
            continue
        X = pd.DataFrame([row[feature_cols]])
        prob = float(model.predict_proba(X)[0][1])
        results.append({
            "date": row_idx,
            "close": row["close"],
            "prob": prob,
            "vix": row.get("vix", 0),
            "symbol": symbol,
            "source": "latest-model",
        })

    return results


def _build_row(d: dict, symbol: str) -> dict:
    """Convert a raw signal dict into a formatted output row."""
    regime = detect_regime(d["vix"]) if d["vix"] > 0 else Regime.NORMAL
    prob = d["prob"]

    if prob > 0.8:
        strength = "very_high"
    elif prob > PROB_BUY_THRESHOLD:
        strength = "high"
    elif prob > 0.4:
        strength = "medium"
    else:
        strength = "low"

    if regime == Regime.STRESS:
        suggestion, position_size = "HOLD", 0.0
    elif prob > 0.8:
        suggestion = "BUY"
        position_size = 1.0 if regime == Regime.NORMAL else 0.5
    elif prob > PROB_BUY_THRESHOLD:
        suggestion = "BUY"
        position_size = 0.8 if regime == Regime.NORMAL else 0.4
    else:
        suggestion, position_size = "HOLD", 0.0

    date_str = str(d["date"])[:10]
    return {
        "date": date_str,
        "symbol": symbol,
        "close": round(d["close"], 2),
        "probability": round(prob, 4),
        "signal_strength": strength,
        "suggestion": suggestion,
        "position_size": position_size,
        "regime": regime.value,
        "vix": round(d["vix"], 2),
        "source": d.get("source", "walk-forward"),
    }


def predict_range(start_date: str, end_date: str) -> list[dict]:
    """Generate daily predictions for all assets over a date range.

    Walk-forward signals are used where available (strictly out-of-sample).
    Recent dates outside walk-forward coverage use the latest trained model.

    Args:
        start_date: ISO date string e.g. "2026-04-01"
        end_date:   ISO date string e.g. "2026-04-13"

    Returns:
        List of dicts sorted by date, one row per (date, asset).
    """
    all_symbols = {**SYMBOLS, **{k: k for k in ETF_SYMBOLS if has_cache(k)}}

    print(f"  Collecting walk-forward signals for {len(all_symbols)} assets...")

    rows = []
    for symbol in all_symbols:
        try:
            df = prepare_data(symbol)
        except Exception as e:
            print(f"  Skipping {symbol}: {e}")
            continue

        # Walk-forward signals (out-of-sample)
        wf_signals = collect_asset_signals(df, symbol)
        covered_dates: set[str] = set()

        for date_str, d in wf_signals.items():
            if date_str[:10] < start_date or date_str[:10] > end_date:
                covered_dates.add(date_str[:10])
                continue
            covered_dates.add(date_str[:10])
            rows.append(_build_row(d, symbol))

        # Extend to recent dates not yet covered by walk-forward
        recent = _collect_recent_signals(symbol, covered_dates, start_date, end_date)
        for d in recent:
            rows.append(_build_row(d, symbol))

    rows.sort(key=lambda x: (x["date"], x["symbol"]))
    return rows


def print_range_report(rows: list[dict], start_date: str, end_date: str) -> None:
    """Print a compact daily signal table for the date range."""
    if not rows:
        print(f"  No signals found for {start_date} ~ {end_date}")
        return

    has_recent = any(r["source"] == "latest-model" for r in rows)

    print(f"\n  {'='*95}")
    print(f"  Historical Signals: {start_date} ~ {end_date}")
    if has_recent:
        print(f"  * dates marked [L] use latest-model (not strictly out-of-sample)")
    print(f"  {'='*95}")
    print(f"\n  {'Date':<12s} {'Symbol':<10s} {'Close':>9s} {'Prob':>6s} {'Signal':<10s} "
          f"{'Suggest':<6s} {'Pos':>5s} {'Regime':<8s} {'VIX':>5s} {'Src'}")
    print(f"  {'-'*90}")

    current_date = None
    for r in rows:
        if r["date"] != current_date:
            if current_date is not None:
                print()
            current_date = r["date"]

        suggest_mark = ">>> " if r["suggestion"] == "BUY" else "    "
        src = "[L]" if r["source"] == "latest-model" else "   "
        print(f"  {r['date']:<12s} {r['symbol']:<10s} {r['close']:>9.2f} "
              f"{r['probability']:>5.1%} {r['signal_strength']:<10s} "
              f"{suggest_mark}{r['suggestion']:<6s} {r['position_size']:>4.0%} "
              f"{r['regime']:<8s} {r['vix']:>5.1f} {src}")

    buy_signals = [r for r in rows if r["suggestion"] == "BUY"]
    dates = sorted(set(r["date"] for r in rows))
    print(f"\n  Summary: {len(dates)} trading days, "
          f"{len(buy_signals)} BUY signals across {len(set(r['symbol'] for r in rows))} assets")
    if buy_signals:
        print(f"  BUY signals:")
        for r in buy_signals:
            src = " [latest-model]" if r["source"] == "latest-model" else ""
            print(f"    {r['date']}  {r['symbol']:<10s}  prob={r['probability']:.1%}  "
                  f"pos={r['position_size']:.0%}  regime={r['regime']}{src}")
    print(f"  {'='*95}")
