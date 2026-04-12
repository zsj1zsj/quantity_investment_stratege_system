"""Sharpe ratio optimization experiment.

Goal: Improve Sharpe from 0.14/0.36 to >0.5 by:
  1. Dynamic exit (replace fixed 25-day hold) - most impactful
  2. Gradual entry (scale into positions)
  3. Combinations

Key insight: low Sharpe is caused by "lumpy" returns from low-frequency
fixed-period trades. Dynamic exits should smooth the equity curve.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE, RISK_FREE_RATE
from model.train import prepare_data, _get_feature_cols
from backtest.cost_model import apply_buy_cost, apply_sell_cost
from strategy.regime import detect_regime, apply_regime_cap, Regime


def run(df, cfg):
    """Run backtest with configurable exit and entry strategies.

    cfg keys:
      buy_thresh: float - probability threshold for entry
      pos_med: float - position size for prob > buy_thresh
      pos_high: float - position size for prob > 0.8
      stop_loss: float - stop-loss percentage
      use_regime: bool - use VIX regime detection
      # Exit strategy
      exit_mode: str - "fixed", "ma_break", "trailing", "profit_target", "combined"
      hold_days: int - max holding period (used in all modes as max)
      min_hold: int - minimum holding days before dynamic exit triggers
      # MA break exit params
      ma_period: int - MA period for trend break exit (e.g., 10)
      # Trailing stop params
      trail_pct: float - trailing stop percentage (e.g., 0.03 = 3%)
      # Profit target params
      profit_target: float - take profit percentage (e.g., 0.05 = 5%)
      # Combined mode uses all applicable dynamic exits
      # Entry strategy
      gradual: bool - use gradual entry
      gradual_stages: list of (day, fraction) - e.g., [(0, 0.3), (5, 0.6), (10, 1.0)]
    """
    feature_cols = _get_feature_cols()
    n = len(df)

    # Step 1: Walk-forward to collect out-of-sample predictions
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
                "trend_up": row.get("close_ma20_ratio", 1) > 1,
                "vix": row.get("vix", 0),
                "ma5": row.get("ma5", 0),
                "ma10": row["close"],  # will compute inline
                "ma20": row.get("ma20", 0),
            })
        start += STEP_SIZE

    # Deduplicate
    seen = set()
    unique = []
    for d in all_data:
        k = str(d["date"])
        if k not in seen:
            seen.add(k)
            unique.append(d)
    unique.sort(key=lambda x: x["date"])
    if not unique:
        return None

    # Pre-compute MA10 from close prices
    closes = np.array([d["close"] for d in unique])
    ma_period = cfg.get("ma_period", 10)
    ma_vals = pd.Series(closes).rolling(window=ma_period).mean().values
    for i, d in enumerate(unique):
        d["ma_dynamic"] = ma_vals[i] if not np.isnan(ma_vals[i]) else d["close"]

    # Step 2: Simulate
    buy_thresh = cfg["buy_thresh"]
    pos_med = cfg["pos_med"]
    pos_high = cfg["pos_high"]
    stop_loss = cfg["stop_loss"]
    use_regime = cfg["use_regime"]
    exit_mode = cfg["exit_mode"]
    hold_days_max = cfg["hold_days"]
    min_hold = cfg.get("min_hold", 5)
    trail_pct = cfg.get("trail_pct", 0.03)
    profit_target = cfg.get("profit_target", 0.05)
    gradual = cfg.get("gradual", False)
    gradual_stages = cfg.get("gradual_stages", [(0, 0.3), (5, 0.6), (10, 1.0)])

    portfolio = 1.0
    in_pos = False
    entry_price = 0.0
    entry_idx = 0
    pos_size = 0.0
    peak_price = 0.0  # for trailing stop
    trade_rets = []
    values = [1.0]

    for i, d in enumerate(unique):
        close = d["close"]
        regime = detect_regime(d["vix"]) if use_regime and d["vix"] > 0 else Regime.NORMAL

        if not in_pos:
            # Entry
            if regime == Regime.STRESS:
                values.append(portfolio)
                continue

            if d["prob"] > buy_thresh:
                entry_price = apply_buy_cost(close)
                entry_idx = i
                raw_size = pos_high if d["prob"] > 0.8 else pos_med
                target_size = apply_regime_cap(raw_size, regime) if use_regime else raw_size

                if gradual:
                    # Start with first stage fraction
                    pos_size = target_size * gradual_stages[0][1]
                else:
                    pos_size = target_size
                peak_price = close
                in_pos = True
        else:
            days_held = i - entry_idx
            dd = max(0, (entry_price - close) / entry_price)
            peak_price = max(peak_price, close)
            trail_dd = (peak_price - close) / peak_price if peak_price > 0 else 0

            # Gradual position building
            if gradual:
                raw_size = pos_high if d["prob"] > 0.8 else pos_med
                target_size = apply_regime_cap(raw_size, regime) if use_regime else raw_size
                for stage_day, stage_frac in gradual_stages:
                    if days_held == stage_day and stage_day > 0 and d["prob"] > 0.4:
                        new_size = target_size * stage_frac
                        if new_size > pos_size:
                            pos_size = new_size

            # Exit logic
            should_sell = False
            exit_reason = ""

            # Always check stop-loss
            if dd > stop_loss:
                should_sell = True
                exit_reason = "stop_loss"

            # Always check max hold
            elif days_held >= hold_days_max:
                should_sell = True
                exit_reason = "max_hold"

            # Dynamic exits (only after min_hold)
            elif days_held >= min_hold:
                if exit_mode == "ma_break":
                    # Exit when close breaks below MA
                    if close < d["ma_dynamic"]:
                        should_sell = True
                        exit_reason = "ma_break"

                elif exit_mode == "trailing":
                    # Trailing stop from peak
                    if trail_dd > trail_pct:
                        should_sell = True
                        exit_reason = "trailing"

                elif exit_mode == "profit_target":
                    # Take profit
                    gain = (close - entry_price) / entry_price
                    if gain > profit_target:
                        should_sell = True
                        exit_reason = "profit_target"

                elif exit_mode == "combined":
                    # All dynamic exits
                    gain = (close - entry_price) / entry_price
                    if close < d["ma_dynamic"]:
                        should_sell = True
                        exit_reason = "ma_break"
                    elif trail_dd > trail_pct:
                        should_sell = True
                        exit_reason = "trailing"
                    elif gain > profit_target:
                        should_sell = True
                        exit_reason = "profit_target"

                # else: exit_mode == "fixed", only max_hold or stop_loss

            if should_sell:
                exit_price = apply_sell_cost(close)
                ret = exit_price / entry_price - 1
                trade_rets.append(ret)
                portfolio *= (1 + ret * pos_size)
                in_pos = False

        values.append(portfolio)

    # Close open position
    if in_pos and unique:
        exit_price = apply_sell_cost(unique[-1]["close"])
        ret = exit_price / entry_price - 1
        trade_rets.append(ret)
        portfolio *= (1 + ret * pos_size)
        values[-1] = portfolio

    vals = np.array(values)
    years = len(vals) / 252
    if years <= 0 or not trade_rets:
        return None

    ar = (vals[-1] / vals[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(vals)
    dd = float(np.max((peak - vals) / peak))
    dr = np.diff(vals) / vals[:-1]
    vol = np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 1
    sr = (ar - RISK_FREE_RATE) / vol
    nt = len(trade_rets)
    wr = sum(1 for r in trade_rets if r > 0) / nt if nt else 0
    wins = sum(r for r in trade_rets if r > 0)
    losses = abs(sum(r for r in trade_rets if r <= 0))
    pf = wins / losses if losses > 0 else float("inf")

    return {
        "ar": ar, "dd": dd, "sr": sr, "vol": vol,
        "nt": nt, "nt_yr": nt / years, "wr": wr, "pf": pf,
    }


# ---- Experiment configurations ----

configs = []

# Baseline: current strategy (fixed 25-day hold)
configs.append(("BASELINE fixed 25d", {
    "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
    "stop_loss": 0.10, "use_regime": True,
    "exit_mode": "fixed", "hold_days": 25,
}))

# === A. Dynamic exit: MA break ===
for ma_p in [5, 10, 15]:
    for min_h in [3, 5, 7]:
        configs.append((f"MA{ma_p} break, minH={min_h}", {
            "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
            "stop_loss": 0.10, "use_regime": True,
            "exit_mode": "ma_break", "hold_days": 30, "min_hold": min_h,
            "ma_period": ma_p,
        }))

# === B. Dynamic exit: Trailing stop ===
for trail in [0.02, 0.03, 0.04, 0.05]:
    for min_h in [3, 5]:
        configs.append((f"Trail {trail:.0%}, minH={min_h}", {
            "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
            "stop_loss": 0.10, "use_regime": True,
            "exit_mode": "trailing", "hold_days": 30, "min_hold": min_h,
            "trail_pct": trail,
        }))

# === C. Dynamic exit: Profit target ===
for target in [0.03, 0.04, 0.05, 0.06]:
    configs.append((f"ProfitTarget {target:.0%}", {
        "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
        "stop_loss": 0.10, "use_regime": True,
        "exit_mode": "profit_target", "hold_days": 30, "min_hold": 5,
        "profit_target": target,
    }))

# === D. Combined dynamic exit ===
for ma_p in [10, 15]:
    for trail in [0.03, 0.04]:
        for target in [0.04, 0.05]:
            configs.append((f"Comb MA{ma_p}/T{trail:.0%}/P{target:.0%}", {
                "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
                "stop_loss": 0.10, "use_regime": True,
                "exit_mode": "combined", "hold_days": 30, "min_hold": 5,
                "ma_period": ma_p, "trail_pct": trail, "profit_target": target,
            }))

# === E. Gradual entry + best dynamic exits ===
for exit_mode in ["ma_break", "trailing", "combined"]:
    cfg = {
        "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
        "stop_loss": 0.10, "use_regime": True,
        "exit_mode": exit_mode, "hold_days": 30, "min_hold": 5,
        "ma_period": 10, "trail_pct": 0.03, "profit_target": 0.05,
        "gradual": True,
        "gradual_stages": [(0, 0.4), (5, 0.7), (10, 1.0)],
    }
    configs.append((f"Gradual + {exit_mode}", cfg))

# === F. Shorter max hold with dynamic ===
for max_h in [15, 20]:
    configs.append((f"MA10 break, maxH={max_h}", {
        "buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
        "stop_loss": 0.10, "use_regime": True,
        "exit_mode": "ma_break", "hold_days": max_h, "min_hold": 5,
        "ma_period": 10,
    }))


def main():
    for symbol in ["SP500", "NASDAQ"]:
        df = prepare_data(symbol)
        print(f"\n{'='*110}")
        print(f"  {symbol} - Sharpe Optimization (Target: SR > 0.5)")
        print(f"{'='*110}")
        print(f"  {'Description':<30s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
        print(f"  {'-'*100}")

        results = []
        for desc, cfg in configs:
            r = run(df, cfg)
            if r is None:
                print(f"  {desc:<30s} | {'N/A':>6s}")
                continue

            gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
            g = "PASS" if gate2 else ""
            marker = " ***" if r["sr"] > 0.5 else (" *" if r["sr"] > 0.3 else "")
            print(f"  {desc:<30s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{marker}")
            results.append((desc, r))

        # Summary: top 5 by Sharpe
        results.sort(key=lambda x: x[1]["sr"], reverse=True)
        print(f"\n  Top 5 by Sharpe for {symbol}:")
        for desc, r in results[:5]:
            print(f"    {desc:<30s} | AR={r['ar']:.2%} | DD={r['dd']:.2%} | SR={r['sr']:.2f} | T/yr={r['nt_yr']:.1f}")


if __name__ == "__main__":
    main()
