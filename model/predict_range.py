"""Date-range historical prediction report.

Generates walk-forward predictions for every trading day in a given date
range, showing what signal the model would have output on each day.
Useful for reviewing historical decision quality or debugging.
"""
from __future__ import annotations

import pandas as pd

from config import SYMBOLS, ETF_SYMBOLS, PROB_BUY_THRESHOLD
from data.store import has_cache
from model.train import prepare_data
from backtest.signals import collect_asset_signals
from strategy.regime import detect_regime, Regime


def predict_range(start_date: str, end_date: str) -> list[dict]:
    """Generate daily predictions for all assets over a date range.

    Uses walk-forward out-of-sample signals — each prediction was made
    using only data available before that date.

    Args:
        start_date: ISO date string e.g. "2025-01-01"
        end_date:   ISO date string e.g. "2025-03-31"

    Returns:
        List of dicts sorted by date, one row per (date, asset).
    """
    # Load all available assets
    prepared = {}
    for key in SYMBOLS:
        try:
            prepared[key] = prepare_data(key)
        except Exception as e:
            print(f"  Skipping {key}: {e}")
    for key in ETF_SYMBOLS:
        if has_cache(key):
            try:
                prepared[key] = prepare_data(key)
            except Exception:
                pass

    print(f"  Collecting walk-forward signals for {len(prepared)} assets...")

    rows = []
    for symbol, df in prepared.items():
        signals = collect_asset_signals(df, symbol)
        for date_str, d in signals.items():
            if date_str[:10] < start_date or date_str[:10] > end_date:
                continue
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
                suggestion = "HOLD"
                position_size = 0.0
            elif prob > 0.8:
                suggestion = "BUY"
                position_size = 1.0 if regime == Regime.NORMAL else 0.5
            elif prob > PROB_BUY_THRESHOLD:
                suggestion = "BUY"
                position_size = 0.8 if regime == Regime.NORMAL else 0.4
            else:
                suggestion = "HOLD"
                position_size = 0.0

            rows.append({
                "date": date_str[:10],
                "symbol": symbol,
                "close": round(d["close"], 2),
                "probability": round(prob, 4),
                "signal_strength": strength,
                "suggestion": suggestion,
                "position_size": position_size,
                "regime": regime.value,
                "vix": round(d["vix"], 2),
            })

    rows.sort(key=lambda x: (x["date"], x["symbol"]))
    return rows


def print_range_report(rows: list[dict], start_date: str, end_date: str) -> None:
    """Print a compact daily signal table for the date range."""
    if not rows:
        print(f"  No signals found for {start_date} ~ {end_date}")
        print("  (Date range may be outside walk-forward coverage)")
        return

    print(f"\n  {'='*90}")
    print(f"  Historical Signals: {start_date} ~ {end_date}")
    print(f"  {'='*90}")
    print(f"\n  {'Date':<12s} {'Symbol':<10s} {'Close':>9s} {'Prob':>6s} {'Signal':<10s} "
          f"{'Suggest':<6s} {'Pos':>5s} {'Regime':<8s} {'VIX':>5s}")
    print(f"  {'-'*85}")

    current_date = None
    for r in rows:
        if r["date"] != current_date:
            if current_date is not None:
                print()
            current_date = r["date"]

        suggest_mark = ">>> " if r["suggestion"] == "BUY" else "    "
        print(f"  {r['date']:<12s} {r['symbol']:<10s} {r['close']:>9.2f} "
              f"{r['probability']:>5.1%} {r['signal_strength']:<10s} "
              f"{suggest_mark}{r['suggestion']:<6s} {r['position_size']:>4.0%} "
              f"{r['regime']:<8s} {r['vix']:>5.1f}")

    # Summary
    buy_signals = [r for r in rows if r["suggestion"] == "BUY"]
    dates = sorted(set(r["date"] for r in rows))
    print(f"\n  Summary: {len(dates)} trading days, "
          f"{len(buy_signals)} BUY signals across {len(set(r['symbol'] for r in rows))} assets")
    if buy_signals:
        print(f"  BUY signals:")
        for r in buy_signals:
            print(f"    {r['date']}  {r['symbol']:<10s}  prob={r['probability']:.1%}  "
                  f"pos={r['position_size']:.0%}  regime={r['regime']}")
    print(f"  {'='*90}")
