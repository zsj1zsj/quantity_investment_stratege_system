"""Multi-ETF portfolio parameter search.

Find optimal total exposure, per-asset cap, and asset selection
for 10-asset portfolio targeting SR>0.5, DD<25%.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lightgbm import LGBMClassifier

from config import (
    LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE,
    RISK_FREE_RATE, PROB_BUY_THRESHOLD, STOP_LOSS_PCT,
    HOLD_PERIOD, POSITION_HIGH_CONF, POSITION_MED_CONF,
)
from model.train import prepare_data, _get_feature_cols
from data.store import has_cache
from backtest.cost_model import apply_buy_cost, apply_sell_cost
from strategy.regime import detect_regime, apply_regime_cap, Regime


def collect_signals(prepared_dfs):
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


def run(asset_signals, symbols, per_asset, max_exposure, stop_loss=0.10):
    signals = {s: asset_signals[s] for s in symbols if s in asset_signals}
    all_dates = sorted(set().union(*(s.keys() for s in signals.values())))
    if not all_dates:
        return None

    portfolio = 1.0
    positions = {}
    trade_rets = []
    values = [1.0]

    for i, date_str in enumerate(all_dates):
        day_pnl = 0.0
        to_close = []
        for sym, pos in positions.items():
            if date_str not in signals.get(sym, {}):
                continue
            d = signals[sym][date_str]
            close = d["close"]
            days_held = i - pos["entry_idx"]
            dd = max(0, (pos["entry_price"] - close) / pos["entry_price"])
            if days_held >= HOLD_PERIOD or dd > stop_loss:
                exit_price = apply_sell_cost(close)
                ret = exit_price / pos["entry_price"] - 1
                trade_rets.append(ret)
                day_pnl += ret * pos["size"]
                to_close.append(sym)
        for sym in to_close:
            del positions[sym]

        candidates = []
        for sym, sig in signals.items():
            if sym in positions or date_str not in sig:
                continue
            d = sig[date_str]
            regime = detect_regime(d["vix"]) if d["vix"] > 0 else Regime.NORMAL
            if regime == Regime.STRESS:
                continue
            if d["prob"] > PROB_BUY_THRESHOLD:
                candidates.append((d["prob"], sym, d, regime))

        candidates.sort(reverse=True)
        current_exp = sum(p["size"] for p in positions.values())
        for prob, sym, d, regime in candidates:
            raw_size = POSITION_HIGH_CONF if prob > 0.8 else POSITION_MED_CONF
            size = apply_regime_cap(raw_size, regime)
            size = min(size, per_asset)
            if current_exp + size > max_exposure:
                size = max_exposure - current_exp
                if size <= 0.05:
                    continue
            entry_price = apply_buy_cost(d["close"])
            positions[sym] = {"entry_price": entry_price, "entry_idx": i, "size": size}
            current_exp += size

        portfolio *= (1 + day_pnl)
        values.append(portfolio)

    for sym, pos in positions.items():
        last = all_dates[-1]
        if last in signals.get(sym, {}):
            d = signals[sym][last]
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
    return {"ar": ar, "dd": dd, "sr": sr, "vol": vol, "nt": nt, "nt_yr": nt/years, "wr": wr}


def main():
    from config import SYMBOLS, ETF_SYMBOLS

    print("Loading data...")
    prepared = {}
    for key in SYMBOLS:
        prepared[key] = prepare_data(key)
    for key in ETF_SYMBOLS:
        if has_cache(key):
            try:
                prepared[key] = prepare_data(key)
            except Exception:
                pass
    print(f"Assets: {list(prepared.keys())}")

    print("Collecting signals...")
    asset_signals = collect_signals(prepared)

    all_syms = list(prepared.keys())
    # Exclude XLE (worst performer: DD=33%, SR=-0.30)
    no_xle = [s for s in all_syms if s != "XLE"]
    # Only good performers (SR > 0 individually)
    good = [s for s in all_syms if s not in ["XLE", "XLP"]]

    print(f"\n{'='*110}")
    print(f"  Multi-ETF Portfolio Search — Target: SR>0.5, DD<25%, T/yr<50")
    print(f"{'='*110}")

    configs = []

    # Section 1: Total exposure sweep (all 10 assets)
    for exp in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        for pa in [0.3, 0.4, 0.5, 0.6]:
            configs.append((f"All10 exp={exp:.0%} pa={pa:.0%}", all_syms, pa, exp))

    # Section 2: Exclude XLE
    for exp in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        for pa in [0.3, 0.4, 0.5]:
            configs.append((f"No-XLE exp={exp:.0%} pa={pa:.0%}", no_xle, pa, exp))

    # Section 3: Only good performers
    for exp in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for pa in [0.3, 0.4, 0.5]:
            configs.append((f"Good8 exp={exp:.0%} pa={pa:.0%}", good, pa, exp))

    # Section 4: Original 2 assets (baseline)
    configs.append(("SP500+NASDAQ pa=60%", ["SP500", "NASDAQ"], 0.6, 1.2))

    print(f"\n  {'Description':<30s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | Gate2")
    print(f"  {'-'*95}")

    results = []
    for desc, syms, pa, exp in configs:
        r = run(asset_signals, syms, pa, exp)
        if r is None:
            continue
        gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
        g = "PASS" if gate2 else ""
        marker = " ***" if gate2 else (" *" if r["sr"] > 0.5 else "")
        print(f"  {desc:<30s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {g}{marker}")
        results.append((desc, r, gate2))

    passed = [(d, r) for d, r, g in results if g]
    print(f"\n  === PASSED Gate 2 ({len(passed)}): ===")
    if passed:
        passed.sort(key=lambda x: x[1]["sr"], reverse=True)
        for desc, r in passed[:10]:
            print(f"    {desc:<30s} | AR={r['ar']:.2%} | DD={r['dd']:.2%} | SR={r['sr']:.2f} | T/yr={r['nt_yr']:.1f}")
    else:
        print("    None. Closest:")
        results.sort(key=lambda x: abs(x[1]["sr"] - 0.5) + max(0, x[1]["dd"] - 0.25) * 5)
        for desc, r, _ in results[:5]:
            print(f"    {desc:<30s} | AR={r['ar']:.2%} | DD={r['dd']:.2%} | SR={r['sr']:.2f} | T/yr={r['nt_yr']:.1f}")


if __name__ == "__main__":
    main()
