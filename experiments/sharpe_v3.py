"""Sharpe optimization v3: fine-tune multi-asset portfolio.

From v2 findings:
  - Multi-asset (SP500+NASDAQ) is the only approach that reaches SR>0.5
  - 60%/asset: SR=0.51 but DD=27.7% (needs <25%)
  - 50%/asset: SR=0.46, DD=26.3% (close on both)

This experiment fine-tunes to find the sweet spot: SR>0.5 AND DD<25%.
Levers: per-asset size, stop-loss, regime sensitivity, hold period.
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


# Allow overriding VIX thresholds for regime
def detect_regime_custom(vix, caution_thresh=20, stress_thresh=30):
    if vix >= stress_thresh:
        return Regime.STRESS
    if vix >= caution_thresh:
        return Regime.CAUTION
    return Regime.NORMAL


def run_multi(dfs, cfg):
    """Multi-asset backtest with fine-grained controls."""
    feature_cols = _get_feature_cols()

    buy_thresh = cfg["buy_thresh"]
    hold_days = cfg["hold_days"]
    pos_med = cfg["pos_med"]
    pos_high = cfg["pos_high"]
    stop_loss = cfg["stop_loss"]
    per_asset_max = cfg["per_asset_max"]
    vix_caution = cfg.get("vix_caution", 20)
    vix_stress = cfg.get("vix_stress", 30)
    # Portfolio-level drawdown circuit breaker
    portfolio_dd_limit = cfg.get("portfolio_dd_limit", 1.0)  # 1.0 = disabled
    # Trailing stop per trade
    trail_pct = cfg.get("trail_pct", 0)  # 0 = disabled

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

    all_dates = sorted(set().union(*(s.keys() for s in asset_signals.values())))
    if not all_dates:
        return None

    portfolio = 1.0
    peak_portfolio = 1.0
    positions = {}  # symbol -> {entry_price, entry_idx, size, peak_price}
    trade_rets = []
    values = [1.0]
    frozen = False  # portfolio circuit breaker

    for i, date_str in enumerate(all_dates):
        day_pnl = 0.0

        # Portfolio-level circuit breaker check
        port_dd = (peak_portfolio - portfolio) / peak_portfolio if peak_portfolio > 0 else 0
        if port_dd > portfolio_dd_limit:
            frozen = True
        elif port_dd < portfolio_dd_limit * 0.5:
            frozen = False  # unfreeze when recovered

        # Check exits
        to_close = []
        for sym, pos in positions.items():
            if date_str not in asset_signals.get(sym, {}):
                continue
            d = asset_signals[sym][date_str]
            close = d["close"]
            days_held = i - pos["entry_idx"]
            dd = max(0, (pos["entry_price"] - close) / pos["entry_price"])

            # Update peak for trailing stop
            if close > pos.get("peak_price", close):
                pos["peak_price"] = close
            trail_dd = (pos["peak_price"] - close) / pos["peak_price"] if trail_pct > 0 else 0

            should_exit = (days_held >= hold_days
                          or dd > stop_loss
                          or (trail_pct > 0 and trail_dd > trail_pct and days_held >= 5))

            if should_exit:
                exit_price = apply_sell_cost(close)
                ret = exit_price / pos["entry_price"] - 1
                trade_rets.append(ret)
                day_pnl += ret * pos["size"]
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Check entries
        if not frozen:
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
            for prob, sym, d, regime in candidates:
                raw_size = pos_high if prob > 0.8 else pos_med
                size = apply_regime_cap(raw_size, regime)
                size = min(size, per_asset_max)
                entry_price = apply_buy_cost(d["close"])
                positions[sym] = {
                    "entry_price": entry_price, "entry_idx": i,
                    "size": size, "peak_price": d["close"],
                }

        portfolio *= (1 + day_pnl)
        peak_portfolio = max(peak_portfolio, portfolio)
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
    dfs = {}
    for symbol in ["SP500", "NASDAQ"]:
        dfs[symbol] = prepare_data(symbol)

    base = {"buy_thresh": 0.5, "pos_med": 0.8, "pos_high": 1.0,
            "hold_days": 25, "stop_loss": 0.10}

    configs = []

    # === Section 1: Per-asset size sweep (fine grain) ===
    for pam in [0.40, 0.45, 0.50, 0.55, 0.60]:
        c = {**base, "per_asset_max": pam}
        configs.append((f"per_asset={pam:.0%}", c))

    # === Section 2: Tighter stop-loss to control DD ===
    for sl in [0.06, 0.07, 0.08]:
        for pam in [0.50, 0.55, 0.60]:
            c = {**base, "per_asset_max": pam, "stop_loss": sl}
            configs.append((f"SL={sl:.0%}, asset={pam:.0%}", c))

    # === Section 3: More sensitive regime (lower VIX thresholds) ===
    for vc, vs in [(18, 25), (15, 25), (18, 28)]:
        for pam in [0.50, 0.55, 0.60]:
            c = {**base, "per_asset_max": pam, "vix_caution": vc, "vix_stress": vs}
            configs.append((f"VIX {vc}/{vs}, asset={pam:.0%}", c))

    # === Section 4: Portfolio circuit breaker ===
    for pdd in [0.15, 0.18, 0.20]:
        for pam in [0.50, 0.55, 0.60]:
            c = {**base, "per_asset_max": pam, "portfolio_dd_limit": pdd}
            configs.append((f"portDD={pdd:.0%}, asset={pam:.0%}", c))

    # === Section 5: Trailing stop per trade (reduces holding losers) ===
    for trail in [0.04, 0.05]:
        for pam in [0.50, 0.55, 0.60]:
            c = {**base, "per_asset_max": pam, "trail_pct": trail}
            configs.append((f"trail={trail:.0%}, asset={pam:.0%}", c))

    # === Section 6: Best combos ===
    combos = [
        ("SL8+VIX18/25, 55%", {**base, "per_asset_max": 0.55, "stop_loss": 0.08, "vix_caution": 18, "vix_stress": 25}),
        ("SL8+VIX18/25, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.08, "vix_caution": 18, "vix_stress": 25}),
        ("SL7+VIX18/25, 55%", {**base, "per_asset_max": 0.55, "stop_loss": 0.07, "vix_caution": 18, "vix_stress": 25}),
        ("SL7+VIX18/25, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.07, "vix_caution": 18, "vix_stress": 25}),
        ("SL8+portDD18, 55%", {**base, "per_asset_max": 0.55, "stop_loss": 0.08, "portfolio_dd_limit": 0.18}),
        ("SL8+portDD18, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.08, "portfolio_dd_limit": 0.18}),
        ("SL8+VIX18/25+portDD20, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.08,
                                          "vix_caution": 18, "vix_stress": 25, "portfolio_dd_limit": 0.20}),
        ("SL7+VIX15/25+portDD18, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.07,
                                          "vix_caution": 15, "vix_stress": 25, "portfolio_dd_limit": 0.18}),
        ("SL8+trail5+VIX18/25, 55%", {**base, "per_asset_max": 0.55, "stop_loss": 0.08,
                                        "vix_caution": 18, "vix_stress": 25, "trail_pct": 0.05}),
        ("SL8+trail5+VIX18/25, 60%", {**base, "per_asset_max": 0.60, "stop_loss": 0.08,
                                        "vix_caution": 18, "vix_stress": 25, "trail_pct": 0.05}),
    ]
    configs.extend(combos)

    print(f"\n{'='*115}")
    print(f"  Multi-Asset (SP500+NASDAQ) Fine-Tuning — Target: SR>0.5 AND DD<25%")
    print(f"{'='*115}")
    print(f"  {'Description':<35s} | {'AR':>6s} | {'DD':>6s} | {'SR':>6s} | {'Vol':>6s} | {'T/yr':>5s} | {'WR':>5s} | {'PF':>5s} | Gate2")
    print(f"  {'-'*105}")

    results = []
    for desc, cfg in configs:
        r = run_multi(dfs, cfg)
        if r is None:
            print(f"  {desc:<35s} | N/A")
            continue
        gate2 = r["ar"] > 0.05 and r["dd"] < 0.25 and r["sr"] > 0.5 and r["nt_yr"] < 50
        g = "PASS" if gate2 else ""
        sr_mark = " ***" if r["sr"] > 0.5 else (" *" if r["sr"] > 0.3 else "")
        dd_mark = " [DD!]" if r["dd"] > 0.25 else ""
        print(f"  {desc:<35s} | {r['ar']:>5.2%} | {r['dd']:>5.2%} | {r['sr']:>6.2f} | {r['vol']:>5.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>4.1%} | {r['pf']:>5.2f} | {g}{sr_mark}{dd_mark}")
        results.append((desc, r, gate2))

    # Summary
    passed = [(d, r) for d, r, g in results if g]
    print(f"\n  === PASSED Gate 2 ({len(passed)} configs): ===")
    if passed:
        for desc, r in passed:
            print(f"    {desc:<35s} | AR={r['ar']:.2%} | DD={r['dd']:.2%} | SR={r['sr']:.2f} | T/yr={r['nt_yr']:.1f}")
    else:
        print("    None yet. Closest:")
        # Show top 5 by distance to gate
        def gate_distance(r):
            d = 0
            if r["sr"] <= 0.5: d += (0.5 - r["sr"])
            if r["dd"] >= 0.25: d += (r["dd"] - 0.25)
            if r["ar"] <= 0.05: d += (0.05 - r["ar"])
            return d
        results.sort(key=lambda x: gate_distance(x[1]))
        for desc, r, _ in results[:8]:
            print(f"    {desc:<35s} | AR={r['ar']:.2%} | DD={r['dd']:.2%} | SR={r['sr']:.2f} | T/yr={r['nt_yr']:.1f}")


if __name__ == "__main__":
    main()
