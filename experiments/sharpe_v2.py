"""Sharpe optimization v2: overlapping positions + multi-asset.

Key insight from v1 experiment: dynamic exits don't help because trades
are already few. The real problem is capital sitting idle most of the time.

Approaches:
  A. Allow overlapping positions (multiple concurrent entries on same asset)
  B. Multi-asset ETFs (more trading opportunities, diversification)
  C. Both combined
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


def run_overlap(df, buy_thresh, hold_days, pos_med, pos_high,
                stop_loss, use_regime, max_positions, cooldown):
    """Backtest allowing overlapping positions on same asset.

    max_positions: max concurrent open positions
    cooldown: min days between entries to avoid clustering
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
                "date": row_idx, "close": row["close"], "prob": probs[idx],
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

    # Track multiple positions
    positions = []  # list of {entry_price, entry_idx, size}
    portfolio = 1.0
    trade_rets = []
    values = [1.0]
    last_entry_idx = -999

    for i, d in enumerate(unique):
        close = d["close"]
        regime = detect_regime(d["vix"]) if use_regime and d["vix"] > 0 else Regime.NORMAL
        day_pnl = 0.0

        # Check exits for all positions
        to_close = []
        for j, pos in enumerate(positions):
            days_held = i - pos["entry_idx"]
            dd = max(0, (pos["entry_price"] - close) / pos["entry_price"])

            if days_held >= hold_days or dd > stop_loss:
                exit_price = apply_sell_cost(close)
                ret = exit_price / pos["entry_price"] - 1
                trade_rets.append(ret)
                day_pnl += ret * pos["size"]
                to_close.append(j)

        for j in reversed(to_close):
            positions.pop(j)

        # Check entry
        if (len(positions) < max_positions
            and regime != Regime.STRESS
            and d["prob"] > buy_thresh
            and (i - last_entry_idx) >= cooldown):

            entry_price = apply_buy_cost(close)
            raw_size = pos_high if d["prob"] > 0.8 else pos_med
            size = apply_regime_cap(raw_size, regime) if use_regime else raw_size
            # Scale down position by max_positions to keep total exposure ~100%
            size = size / max_positions
            positions.append({"entry_price": entry_price, "entry_idx": i, "size": size})
            last_entry_idx = i

        portfolio *= (1 + day_pnl)
        values.append(portfolio)

    # Close remaining
    for pos in positions:
        exit_price = apply_sell_cost(unique[-1]["close"])
        ret = exit_price / pos["entry_price"] - 1
        trade_rets.append(ret)
        portfolio *= (1 + ret * pos["size"])
    if positions:
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

    return {"ar": ar, "dd": dd, "sr": sr, "vol": vol, "nt": nt, "nt_yr": nt/years, "wr": wr, "pf": pf}


def run_multi_asset(dfs, buy_thresh, hold_days, pos_med, pos_high,
                    stop_loss, use_regime, per_asset_max):
    """Backtest across multiple assets with shared portfolio.

    dfs: dict of {symbol: prepared_dataframe}
    per_asset_max: max position per asset as fraction of total portfolio
    """
    feature_cols = _get_feature_cols()

    # Collect signals for each asset
    asset_signals = {}
    for symbol, df in dfs.items():
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
                    "vix": row.get("vix", 0), "symbol": symbol,
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

    # Get union of all dates
    all_dates = set()
    for signals in asset_signals.values():
        all_dates.update(signals.keys())
    all_dates = sorted(all_dates)
    if not all_dates:
        return None

    # Simulate
    portfolio = 1.0
    positions = {}  # symbol -> {entry_price, entry_idx, size}
    trade_rets = []
    values = [1.0]

    for i, date_str in enumerate(all_dates):
        day_pnl = 0.0

        # Check exits
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
                trade_rets.append(ret)
                day_pnl += ret * pos["size"]
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Check entries (prioritize highest prob)
        candidates = []
        for sym, signals in asset_signals.items():
            if sym in positions or date_str not in signals:
                continue
            d = signals[date_str]
            regime = detect_regime(d["vix"]) if use_regime and d["vix"] > 0 else Regime.NORMAL
            if regime == Regime.STRESS:
                continue
            if d["prob"] > buy_thresh:
                candidates.append((d["prob"], sym, d, regime))

        candidates.sort(reverse=True)  # highest prob first
        for prob, sym, d, regime in candidates:
            raw_size = pos_high if prob > 0.8 else pos_med
            size = apply_regime_cap(raw_size, regime) if use_regime else raw_size
            size = min(size, per_asset_max)  # cap per asset
            entry_price = apply_buy_cost(d["close"])
            positions[sym] = {"entry_price": entry_price, "entry_idx": i, "size": size}

        portfolio *= (1 + day_pnl)
        values.append(portfolio)

    # Close remaining
    for sym, pos in positions.items():
        last_date = all_dates[-1]
        if last_date in asset_signals.get(sym, {}):
            d = asset_signals[sym][last_date]
            exit_price = apply_sell_cost(d["close"])
            ret = exit_price / pos["entry_price"] - 1
            trade_rets.append(ret)
            portfolio *= (1 + ret * pos["size"])
    if positions:
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

    return {"ar": ar, "dd": dd, "sr": sr, "vol": vol, "nt": nt, "nt_yr": nt/years, "wr": wr, "pf": pf}


def main():
    # ==========================================
    # Part A: Overlapping positions on same asset
    # ==========================================
    print("\n" + "=" * 110)
    print("  PART A: Overlapping Positions")
    print("=" * 110)

    for symbol in ["SP500", "NASDAQ"]:
        df = prepare_data(symbol)
        print(f"\n  {symbol}:")
        print(f"  {'Description':<35s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
        print(f"  {'-'*105}")

        configs = [
            # (max_pos, cooldown, desc)
            (1, 0,  "Baseline (1 pos)"),
            (2, 3,  "2 pos, 3d cooldown"),
            (2, 5,  "2 pos, 5d cooldown"),
            (2, 10, "2 pos, 10d cooldown"),
            (3, 3,  "3 pos, 3d cooldown"),
            (3, 5,  "3 pos, 5d cooldown"),
            (3, 10, "3 pos, 10d cooldown"),
        ]

        for max_pos, cd, desc in configs:
            r = run_overlap(df, 0.5, 25, 0.8, 1.0, 0.10, True, max_pos, cd)
            if r is None:
                print(f"  {desc:<35s} | N/A")
                continue
            gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
            g = "PASS" if gate2 else ""
            marker = " ***" if r["sr"] > 0.5 else (" *" if r["sr"] > 0.3 else "")
            print(f"  {desc:<35s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{marker}")

    # ==========================================
    # Part B: Multi-asset (SP500 + NASDAQ combined)
    # ==========================================
    print("\n" + "=" * 110)
    print("  PART B: Multi-Asset Portfolio (SP500 + NASDAQ)")
    print("=" * 110)

    dfs = {}
    for symbol in ["SP500", "NASDAQ"]:
        dfs[symbol] = prepare_data(symbol)

    print(f"\n  {'Description':<35s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
    print(f"  {'-'*105}")

    multi_configs = [
        # (buy_thresh, hold, pos_med, pos_high, stop_loss, per_asset_max, desc)
        (0.5, 25, 0.8, 1.0, 0.10, 0.5,  "50% per asset"),
        (0.5, 25, 0.8, 1.0, 0.10, 0.6,  "60% per asset"),
        (0.5, 25, 0.8, 1.0, 0.10, 0.7,  "70% per asset"),
        (0.5, 25, 0.8, 1.0, 0.10, 0.8,  "80% per asset"),
        (0.5, 25, 0.8, 1.0, 0.10, 1.0,  "100% per asset (full)"),
        (0.45, 25, 0.8, 1.0, 0.10, 0.5, "thresh=0.45, 50%/asset"),
        (0.5, 20, 0.8, 1.0, 0.10, 0.5,  "hold 20d, 50%/asset"),
    ]

    for bt, hd, pm, ph, sl, pam, desc in multi_configs:
        r = run_multi_asset(dfs, bt, hd, pm, ph, sl, True, pam)
        if r is None:
            print(f"  {desc:<35s} | N/A")
            continue
        gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
        g = "PASS" if gate2 else ""
        marker = " ***" if r["sr"] > 0.5 else (" *" if r["sr"] > 0.3 else "")
        print(f"  {desc:<35s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{marker}")

    # ==========================================
    # Part C: Overlapping positions + lower threshold
    # ==========================================
    print("\n" + "=" * 110)
    print("  PART C: Overlapping + Lower Threshold")
    print("=" * 110)

    for symbol in ["SP500", "NASDAQ"]:
        df = prepare_data(symbol)
        print(f"\n  {symbol}:")
        print(f"  {'Description':<35s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
        print(f"  {'-'*105}")

        configs = [
            # (thresh, max_pos, cooldown, hold, desc)
            (0.45, 2, 5,  25, "0.45, 2pos, cd5, 25d"),
            (0.45, 3, 5,  25, "0.45, 3pos, cd5, 25d"),
            (0.40, 2, 5,  25, "0.40, 2pos, cd5, 25d"),
            (0.40, 3, 5,  25, "0.40, 3pos, cd5, 25d"),
            (0.45, 2, 5,  20, "0.45, 2pos, cd5, 20d"),
            (0.45, 3, 3,  20, "0.45, 3pos, cd3, 20d"),
            (0.50, 2, 5,  20, "0.50, 2pos, cd5, 20d"),
            (0.50, 3, 5,  20, "0.50, 3pos, cd5, 20d"),
        ]

        for thresh, mp, cd, hd, desc in configs:
            r = run_overlap(df, thresh, hd, 0.8, 1.0, 0.10, True, mp, cd)
            if r is None:
                print(f"  {desc:<35s} | N/A")
                continue
            gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
            g = "PASS" if gate2 else ""
            marker = " ***" if r["sr"] > 0.5 else (" *" if r["sr"] > 0.3 else "")
            print(f"  {desc:<35s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{marker}")


if __name__ == "__main__":
    main()
