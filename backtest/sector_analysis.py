"""Per-sector (per-asset) performance analysis.

Runs individual walk-forward backtests for each asset in the portfolio,
reports per-asset metrics (AR, DD, SR, WR, avg trade return), and ranks
assets by contribution quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    RISK_FREE_RATE, PROB_BUY_THRESHOLD, STOP_LOSS_PCT,
    HOLD_PERIOD, POSITION_HIGH_CONF, POSITION_MED_CONF,
)
from backtest.cost_model import apply_buy_cost, apply_sell_cost
from backtest.signals import collect_asset_signals
from strategy.regime import detect_regime, apply_regime_cap, Regime


def _run_single_asset_backtest(signals: list[dict]) -> dict:
    """Run backtest for a single asset with regime and position sizing."""
    if not signals:
        return None

    portfolio = 1.0
    position = None
    trades = []
    values = [1.0]

    for i, d in enumerate(signals):
        day_pnl = 0.0

        # Check exit
        if position is not None:
            close = d["close"]
            days_held = i - position["entry_idx"]
            dd = max(0, (position["entry_price"] - close) / position["entry_price"])
            if days_held >= HOLD_PERIOD or dd > STOP_LOSS_PCT:
                exit_price = apply_sell_cost(close)
                ret = exit_price / position["entry_price"] - 1
                trades.append(ret)
                day_pnl += ret * position["size"]
                position = None

        # Check entry
        if position is None:
            regime = detect_regime(d["vix"]) if d["vix"] > 0 else Regime.NORMAL
            if regime != Regime.STRESS and d["prob"] > PROB_BUY_THRESHOLD:
                raw_size = POSITION_HIGH_CONF if d["prob"] > 0.8 else POSITION_MED_CONF
                size = apply_regime_cap(raw_size, regime)
                entry_price = apply_buy_cost(d["close"])
                position = {"entry_price": entry_price, "entry_idx": i, "size": size}

        portfolio *= (1 + day_pnl)
        values.append(portfolio)

    # Close remaining position
    if position is not None and signals:
        d = signals[-1]
        exit_price = apply_sell_cost(d["close"])
        ret = exit_price / position["entry_price"] - 1
        trades.append(ret)
        portfolio *= (1 + ret * position["size"])
        values[-1] = portfolio

    vals = np.array(values)
    years = len(vals) / 252
    if years <= 0 or not trades:
        return None

    ar = (vals[-1] / vals[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(vals)
    dd = float(np.max((peak - vals) / peak))
    dr = np.diff(vals) / vals[:-1]
    vol = np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 1
    sr = (ar - RISK_FREE_RATE) / vol
    nt = len(trades)
    wr = sum(1 for r in trades if r > 0) / nt if nt else 0
    avg_ret = np.mean(trades) if trades else 0
    avg_win = np.mean([r for r in trades if r > 0]) if any(r > 0 for r in trades) else 0
    avg_loss = np.mean([r for r in trades if r <= 0]) if any(r <= 0 for r in trades) else 0

    return {
        "ar": ar, "dd": dd, "sr": sr, "vol": vol,
        "nt": nt, "nt_yr": nt / years, "wr": wr,
        "avg_ret": avg_ret, "avg_win": avg_win, "avg_loss": avg_loss,
        "years": years,
    }


def run_sector_analysis(prepared_dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Run per-asset analysis and return sorted results."""
    results = []
    for symbol, df in prepared_dfs.items():
        print(f"  Analyzing {symbol}...")
        signals_dict = collect_asset_signals(df, symbol)
        signals = sorted(signals_dict.values(), key=lambda x: x["date"])
        metrics = _run_single_asset_backtest(signals)
        if metrics is not None:
            metrics["symbol"] = symbol
            results.append(metrics)

    # Sort by Sharpe ratio descending
    results.sort(key=lambda x: x["sr"], reverse=True)
    return results


def print_sector_analysis(results: list[dict]) -> None:
    """Print formatted sector analysis report."""
    print(f"\n{'='*90}")
    print(f"  Per-Asset Performance Analysis (Walk-Forward, Net of Costs)")
    print(f"{'='*90}")

    if not results:
        print("  No results.")
        return

    print(f"\n  {'Asset':<8s} | {'AR':>7s} | {'DD':>7s} | {'SR':>7s} | {'Vol':>7s} | "
          f"{'T/yr':>5s} | {'WR':>6s} | {'AvgRet':>7s} | {'AvgWin':>7s} | {'AvgLoss':>8s} | Grade")
    print(f"  {'-'*100}")

    for r in results:
        # Grade based on SR and DD
        if r["sr"] > 0.5 and r["dd"] < 0.25:
            grade = "A"
        elif r["sr"] > 0 and r["dd"] < 0.25:
            grade = "B"
        elif r["sr"] > 0:
            grade = "C"
        else:
            grade = "D"

        print(f"  {r['symbol']:<8s} | {r['ar']:>6.2%} | {r['dd']:>6.2%} | {r['sr']:>7.2f} | "
              f"{r['vol']:>6.2%} | {r['nt_yr']:>5.1f} | {r['wr']:>5.1%} | "
              f"{r['avg_ret']:>6.2%} | {r['avg_win']:>6.2%} | {r['avg_loss']:>7.2%} | {grade}")

    # Summary statistics
    print(f"\n  Summary:")
    srs = [r["sr"] for r in results]
    ars = [r["ar"] for r in results]
    grade_a = sum(1 for r in results if r["sr"] > 0.5 and r["dd"] < 0.25)
    grade_d = sum(1 for r in results if r["sr"] <= 0)
    print(f"    Assets analyzed: {len(results)}")
    print(f"    SR range: {min(srs):.2f} ~ {max(srs):.2f} (mean={np.mean(srs):.2f})")
    print(f"    AR range: {min(ars):.2%} ~ {max(ars):.2%} (mean={np.mean(ars):.2%})")
    print(f"    Grade A (SR>0.5, DD<25%): {grade_a}")
    print(f"    Grade D (SR<=0): {grade_d}")

    # Recommendation
    if grade_d > 0:
        weak = [r["symbol"] for r in results if r["sr"] <= 0]
        print(f"\n  Recommendation: Consider excluding weak performers: {', '.join(weak)}")
    print(f"{'='*90}")
