"""Rolling stability analysis and parameter sensitivity testing.

Provides:
  1. Rolling 1-year Sharpe ratio and max drawdown
  2. Per-year breakdown of strategy performance
  3. Parameter sensitivity grid test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import (
    LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE,
    RISK_FREE_RATE, PROB_BUY_THRESHOLD, STOP_LOSS_PCT,
    HOLD_PERIOD, POSITION_HIGH_CONF, POSITION_MED_CONF,
)
from model.train import _get_feature_cols
from backtest.cost_model import apply_buy_cost, apply_sell_cost
from strategy.regime import Regime


def detect_regime_custom(vix, caution=18, stress=25):
    if vix >= stress:
        return Regime.STRESS
    if vix >= caution:
        return Regime.CAUTION
    return Regime.NORMAL


def regime_cap(size, regime):
    caps = {Regime.NORMAL: 1.0, Regime.CAUTION: 0.5, Regime.STRESS: 0.0}
    return min(size, caps[regime])


def _collect_all_signals(prepared_dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Walk-forward signal collection for all assets."""
    feature_cols = _get_feature_cols()
    asset_signals = {}

    for symbol, df in prepared_dfs.items():
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
                    "date": row_idx, "close": row["close"],
                    "prob": probs[idx], "vix": row.get("vix", 0),
                    "symbol": symbol,
                })
            start += STEP_SIZE

        seen = set()
        unique = []
        for d in all_data:
            k = str(d["date"])
            if k not in seen:
                seen.add(k)
                unique.append(d)
        unique.sort(key=lambda x: x["date"])
        asset_signals[symbol] = {str(d["date"]): d for d in unique}

    return asset_signals


def _run_multi_backtest(asset_signals, buy_thresh, hold_days, pos_med, pos_high,
                        stop_loss, per_asset_max, vix_caution, vix_stress,
                        max_exposure=None):
    """Run multi-asset backtest, return daily portfolio values with dates."""
    from config import MAX_TOTAL_EXPOSURE
    if max_exposure is None:
        max_exposure = MAX_TOTAL_EXPOSURE

    all_dates = sorted(set().union(*(s.keys() for s in asset_signals.values())))
    if not all_dates:
        return [], []

    portfolio = 1.0
    positions = {}
    values = []
    dates = []

    for i, date_str in enumerate(all_dates):
        day_pnl = 0.0

        to_close = []
        for sym, pos in positions.items():
            if date_str not in asset_signals.get(sym, {}):
                continue
            d = asset_signals[sym][date_str]
            close = d["close"]
            days_held = i - pos["entry_idx"]
            dd = max(0, (pos["entry_price"] - close) / pos["entry_price"])
            if days_held >= hold_days or dd > stop_loss:
                exit_price = apply_sell_cost(close)
                ret = exit_price / pos["entry_price"] - 1
                day_pnl += ret * pos["size"]
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        candidates = []
        for sym, signals in asset_signals.items():
            if sym in positions or date_str not in signals:
                continue
            d = signals[date_str]
            regime = detect_regime_custom(d["vix"], vix_caution, vix_stress)
            if regime == Regime.STRESS:
                continue
            if d["prob"] > buy_thresh:
                candidates.append((d["prob"], sym, d, regime))

        candidates.sort(reverse=True)
        current_exp = sum(p["size"] for p in positions.values())
        for prob, sym, d, regime in candidates:
            raw_size = pos_high if prob > 0.8 else pos_med
            size = regime_cap(raw_size, regime)
            size = min(size, per_asset_max)
            if current_exp + size > max_exposure:
                size = max_exposure - current_exp
                if size <= 0.05:
                    continue
            entry_price = apply_buy_cost(d["close"])
            positions[sym] = {"entry_price": entry_price, "entry_idx": i, "size": size}
            current_exp += size

        portfolio *= (1 + day_pnl)
        values.append(portfolio)
        dates.append(date_str)

    # Close remaining
    for sym, pos in positions.items():
        last_date = all_dates[-1]
        if last_date in asset_signals.get(sym, {}):
            d = asset_signals[sym][last_date]
            exit_price = apply_sell_cost(d["close"])
            ret = exit_price / pos["entry_price"] - 1
            portfolio *= (1 + ret * pos["size"])
    if positions and values:
        values[-1] = portfolio

    return dates, values


def compute_metrics(values):
    """Compute AR, DD, SR, vol from portfolio value series."""
    vals = np.array(values)
    if len(vals) < 2:
        return {"ar": 0, "dd": 0, "sr": 0, "vol": 0}
    years = len(vals) / 252
    if years <= 0:
        return {"ar": 0, "dd": 0, "sr": 0, "vol": 0}
    ar = (vals[-1] / vals[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(vals)
    dd = float(np.max((peak - vals) / peak))
    dr = np.diff(vals) / vals[:-1]
    vol = np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 1
    sr = (ar - RISK_FREE_RATE) / vol
    return {"ar": ar, "dd": dd, "sr": sr, "vol": vol}


# =========================================================
# 1. Rolling Analysis
# =========================================================

def rolling_analysis(dates, values, window_days=252):
    """Compute rolling 1-year Sharpe and max drawdown."""
    vals = np.array(values)
    n = len(vals)
    results = []

    for end in range(window_days, n):
        start = end - window_days
        window_vals = vals[start:end + 1]
        m = compute_metrics(window_vals)
        results.append({
            "date": dates[end],
            "rolling_sr": m["sr"],
            "rolling_dd": m["dd"],
            "rolling_ar": m["ar"],
        })

    return results


def yearly_breakdown(dates, values):
    """Compute per-calendar-year performance."""
    df = pd.DataFrame({"date": dates, "value": values})
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    results = []
    for year, group in df.groupby("year"):
        if len(group) < 20:
            continue
        vals = group["value"].values
        m = compute_metrics(vals)
        results.append({"year": year, **m})

    return results


def print_rolling_analysis(dates, values):
    """Print rolling and yearly analysis."""
    print(f"\n  {'='*70}")
    print(f"  Rolling & Yearly Stability Analysis")
    print(f"  {'='*70}")

    # Yearly breakdown
    yearly = yearly_breakdown(dates, values)
    print(f"\n  Per-Year Performance:")
    print(f"  {'Year':<6s} | {'AR':>7s} | {'DD':>7s} | {'SR':>7s} | {'Vol':>7s} | Status")
    print(f"  {'-'*62}")
    for y in yearly:
        status = "OK" if y["sr"] > 0 else "WEAK"
        if y["dd"] > 0.25:
            status = "HIGH DD"
        print(f"  {y['year']:<6d} | {y['ar']:>6.2%} | {y['dd']:>6.2%} | {y['sr']:>7.2f} | {y['vol']:>6.2%} | {status}")

    # Rolling summary
    rolling = rolling_analysis(dates, values)
    if rolling:
        srs = [r["rolling_sr"] for r in rolling]
        dds = [r["rolling_dd"] for r in rolling]
        print(f"\n  Rolling 1Y Sharpe: min={min(srs):.2f}, max={max(srs):.2f}, "
              f"mean={np.mean(srs):.2f}, std={np.std(srs):.2f}")
        print(f"  Rolling 1Y MaxDD:  min={min(dds):.2%}, max={max(dds):.2%}, "
              f"mean={np.mean(dds):.2%}")
        # Count periods where SR < 0
        negative_pct = sum(1 for s in srs if s < 0) / len(srs)
        print(f"  Periods with negative SR: {negative_pct:.1%}")


# =========================================================
# 2. Parameter Sensitivity
# =========================================================

def parameter_sensitivity(asset_signals, base_params=None):
    """Test parameter sensitivity by perturbing one parameter at a time."""
    from config import PER_ASSET_MAX_POSITION, MAX_TOTAL_EXPOSURE
    from strategy.regime import VIX_CAUTION, VIX_STRESS

    if base_params is None:
        base_params = {
            "buy_thresh": PROB_BUY_THRESHOLD,
            "hold_days": HOLD_PERIOD,
            "pos_med": POSITION_MED_CONF,
            "pos_high": POSITION_HIGH_CONF,
            "stop_loss": STOP_LOSS_PCT,
            "per_asset_max": PER_ASSET_MAX_POSITION,
            "vix_caution": VIX_CAUTION,
            "vix_stress": VIX_STRESS,
        }

    # Perturbation grid
    perturbations = {
        "vix_caution": [15, 16, 17, 18, 19, 20, 22],
        "vix_stress":  [22, 23, 25, 27, 28, 30],
        "per_asset_max": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "hold_days":   [15, 20, 25, 30, 35],
        "stop_loss":   [0.06, 0.08, 0.10, 0.12, 0.15],
        "buy_thresh":  [0.40, 0.45, 0.50, 0.55, 0.60],
        "pos_med":     [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    print(f"\n  {'='*70}")
    print(f"  Parameter Sensitivity Analysis")
    print(f"  {'='*70}")
    print(f"  Base: VIX={base_params['vix_caution']}/{base_params['vix_stress']}, "
          f"asset={base_params['per_asset_max']:.0%}, hold={base_params['hold_days']}d, "
          f"SL={base_params['stop_loss']:.0%}, thresh={base_params['buy_thresh']}")

    for param_name, test_values in perturbations.items():
        print(f"\n  --- {param_name} ---")
        print(f"  {'Value':<10s} | {'AR':>7s} | {'DD':>7s} | {'SR':>7s} | {'Vol':>7s} | Gate2")
        print(f"  {'-'*58}")

        for val in test_values:
            params = {**base_params, param_name: val}
            dates, values = _run_multi_backtest(
                asset_signals,
                params["buy_thresh"], params["hold_days"],
                params["pos_med"], params["pos_high"],
                params["stop_loss"], params["per_asset_max"],
                params["vix_caution"], params["vix_stress"],
            )
            if not values:
                continue
            m = compute_metrics(values)
            gate2 = m["ar"] > 0.05 and m["dd"] < 0.25 and m["sr"] > 0.5
            marker = " <-- base" if val == base_params[param_name] else ""
            g = "PASS" if gate2 else ""
            fmt_val = f"{val:.0%}" if isinstance(val, float) and val < 1.5 else str(val)
            print(f"  {fmt_val:<10s} | {m['ar']:>6.2%} | {m['dd']:>6.2%} | {m['sr']:>7.2f} | {m['vol']:>6.2%} | {g}{marker}")
