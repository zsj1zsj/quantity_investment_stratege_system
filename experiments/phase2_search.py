"""Phase 2 strategy parameter search.

Goal: AR > 5%, DD < 25%, Sharpe > 0.5, trades < 50/yr
Now with Regime detection replacing TrendUp for downside protection.
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


def run(df, buy_thresh, hold_days, require_trend, pos_med, pos_high,
        use_regime, gradual, stop_loss):
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
                "date": row_idx, "close": row["close"], "prob": probs[idx],
                "trend_up": row.get("close_ma20_ratio", 1) > 1,
                "vix": row.get("vix", 0),
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
    if not unique:
        return None

    portfolio = 1.0
    in_pos = False
    entry_price = 0.0
    entry_idx = 0
    pos_size = 0.0
    trade_rets = []
    values = [1.0]

    for i, d in enumerate(unique):
        close = d["close"]
        regime = detect_regime(d["vix"]) if use_regime and d["vix"] > 0 else Regime.NORMAL

        if not in_pos:
            # Entry logic
            if regime == Regime.STRESS:
                values.append(portfolio)
                continue

            buy = d["prob"] > buy_thresh
            if require_trend:
                buy = buy and d["trend_up"]
            if buy:
                entry_price = apply_buy_cost(close)
                entry_idx = i
                raw_size = pos_high if d["prob"] > 0.8 else pos_med
                pos_size = apply_regime_cap(raw_size, regime) if use_regime else raw_size
                if gradual:
                    pos_size = min(0.5, pos_size)
                in_pos = True
        else:
            days_held = i - entry_idx
            dd = max(0, (entry_price - close) / entry_price)

            # Gradual add
            if gradual and days_held == 5 and d["prob"] > 0.4:
                raw_size = pos_high if d["prob"] > 0.8 else pos_med
                new_size = apply_regime_cap(raw_size, regime) if use_regime else raw_size
                if new_size > pos_size:
                    pos_size = new_size

            # Exit
            should_sell = days_held >= hold_days or dd > stop_loss
            if should_sell:
                exit_price = apply_sell_cost(close)
                ret = exit_price / entry_price - 1
                trade_rets.append(ret)
                portfolio *= (1 + ret * pos_size)
                in_pos = False

        values.append(portfolio)

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

    return {"ar": ar, "dd": dd, "sr": sr, "nt": nt, "nt_yr": nt/years, "wr": wr, "pf": pf}


configs = [
    # (buy_thresh, hold, trend, pos_med, pos_high, regime, gradual, stop_loss, desc)
    (0.5,  20, True,  0.5, 0.8, True,  True,  0.08, "current"),
    (0.5,  20, False, 0.5, 0.8, True,  False, 0.08, "no trend, regime only"),
    (0.5,  20, False, 0.6, 0.9, True,  False, 0.08, "no trend, 60/90% pos"),
    (0.5,  20, False, 0.7, 0.9, True,  False, 0.08, "no trend, 70/90% pos"),
    (0.5,  20, False, 0.8, 1.0, True,  False, 0.08, "no trend, 80/100% pos"),
    (0.4,  20, False, 0.7, 0.9, True,  False, 0.08, "thresh=0.4, 70/90%"),
    (0.45, 20, False, 0.7, 0.9, True,  False, 0.08, "thresh=0.45, 70/90%"),
    (0.5,  15, False, 0.7, 0.9, True,  False, 0.08, "hold 15d, 70/90%"),
    (0.5,  25, False, 0.7, 0.9, True,  False, 0.08, "hold 25d, 70/90%"),
    (0.5,  20, False, 0.7, 0.9, True,  False, 0.10, "70/90%, SL=10%"),
    (0.5,  20, False, 0.7, 0.9, True,  False, 0.12, "70/90%, SL=12%"),
    (0.5,  20, False, 0.7, 0.9, True,  True,  0.08, "70/90%, gradual"),
    (0.45, 20, False, 0.8, 1.0, True,  False, 0.10, "0.45, 80/100%, SL=10%"),
    (0.4,  20, False, 0.8, 1.0, True,  False, 0.10, "0.4, 80/100%, SL=10%"),
]


def main():
    for symbol in ["SP500", "NASDAQ"]:
        df = prepare_data(symbol)
        print(f"\n{'='*100}")
        print(f"  {symbol} - Phase 2 Strategy Search (Gate 2: AR>5%, DD<25%, SR>0.5, trades<50/yr)")
        print(f"{'='*100}")
        print(f"  {'Description':<25s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
        print(f"  {'-'*88}")

        for bt, hd, tr, pm, ph, rg, gr, sl, desc in configs:
            r = run(df, bt, hd, tr, pm, ph, rg, gr, sl)
            if r is None:
                print(f"  {desc:<25s} | {'N/A':>6s}")
                continue
            gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
            g = "PASS" if gate2 else ""
            marker = " ***" if r["ar"] > 0.05 else ""
            print(f"  {desc:<25s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{marker}")


if __name__ == "__main__":
    main()
