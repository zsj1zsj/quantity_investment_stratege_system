"""Experiment: test different strategy parameters to find profitable configuration.

Tests combinations of:
  - buy threshold (prob_up > X)
  - with/without TrendUp filter
  - fixed holding period (match 20d label) vs signal-based exit
  - with/without EWM smoothing in backtest
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from dataclasses import dataclass

from config import (
    LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE,
    RISK_FREE_RATE, FORWARD_DAYS,
)
from model.train import prepare_data, _get_feature_cols
from backtest.cost_model import apply_buy_cost, apply_sell_cost


@dataclass
class SimpleResult:
    annual_return: float
    max_drawdown: float
    sharpe: float
    total_trades: int
    win_rate: float
    profit_factor: float


def run_simple_backtest(
    df: pd.DataFrame,
    buy_thresh: float,
    require_trend_up: bool,
    hold_days: int,       # fixed holding period; 0 = use sell signal
    sell_thresh: float,
    smooth: bool,
) -> SimpleResult:
    """Simplified backtest for parameter search."""
    feature_cols = _get_feature_cols()
    n = len(df)

    # Walk-forward: collect all out-of-sample predictions
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
        return SimpleResult(0, 0, 0, 0, 0, 0)

    # Optional smoothing
    if smooth:
        probs_series = pd.Series([d["prob"] for d in unique])
        smoothed = probs_series.ewm(span=3, adjust=False).mean().values
        for i, d in enumerate(unique):
            d["prob"] = smoothed[i]

    # Simulate
    portfolio = 1.0
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    trades_returns = []
    portfolio_values = [1.0]

    for i, d in enumerate(unique):
        if not in_position:
            buy_signal = d["prob"] > buy_thresh
            if require_trend_up:
                buy_signal = buy_signal and d["trend_up"]
            if buy_signal:
                entry_price = apply_buy_cost(d["close"])
                entry_idx = i
                in_position = True
        else:
            days_held = i - entry_idx
            if hold_days > 0:
                # Fixed holding period exit
                should_sell = days_held >= hold_days
            else:
                # Signal-based exit
                should_sell = d["prob"] < sell_thresh or days_held > 40

            # Stop loss always active
            drawdown = max(0, (entry_price - d["close"]) / entry_price)
            if drawdown > 0.08:
                should_sell = True

            if should_sell:
                exit_price = apply_sell_cost(d["close"])
                ret = exit_price / entry_price - 1
                trades_returns.append(ret)
                portfolio *= (1 + ret * 0.5)  # 50% position
                in_position = False

        portfolio_values.append(portfolio)

    # Close open position
    if in_position and unique:
        exit_price = apply_sell_cost(unique[-1]["close"])
        ret = exit_price / entry_price - 1
        trades_returns.append(ret)
        portfolio *= (1 + ret * 0.5)
        portfolio_values[-1] = portfolio

    # Metrics
    values = np.array(portfolio_values)
    n_days = len(values)
    years = n_days / 252
    if years <= 0 or len(trades_returns) == 0:
        return SimpleResult(0, 0, 0, 0, 0, 0)

    annual_ret = (values[-1] / values[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / peak))
    daily_rets = np.diff(values) / values[:-1]
    annual_vol = np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 1
    sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol

    wins = [r for r in trades_returns if r > 0]
    losses = [r for r in trades_returns if r <= 0]
    win_rate = len(wins) / len(trades_returns) if trades_returns else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return SimpleResult(annual_ret, max_dd, sharpe, len(trades_returns), win_rate, pf)


def main():
    configs = [
        # (buy_thresh, require_trend, hold_days, sell_thresh, smooth, desc)
        (0.7, True,  0,  0.3, True,  "baseline (current)"),
        (0.7, True,  0,  0.3, False, "no smoothing"),
        (0.7, False, 0,  0.3, False, "no trend filter"),
        (0.6, False, 0,  0.3, False, "thresh=0.6, no trend"),
        (0.5, False, 0,  0.3, False, "thresh=0.5, no trend"),
        (0.7, False, 20, 0.3, False, "hold 20d"),
        (0.6, False, 20, 0.3, False, "thresh=0.6, hold 20d"),
        (0.5, False, 20, 0.3, False, "thresh=0.5, hold 20d"),
        (0.5, True,  20, 0.3, False, "thresh=0.5, trend, hold 20d"),
        (0.6, True,  20, 0.3, False, "thresh=0.6, trend, hold 20d"),
        (0.5, False, 15, 0.3, False, "thresh=0.5, hold 15d"),
        (0.4, False, 20, 0.3, False, "thresh=0.4, hold 20d"),
    ]

    for symbol in ["SP500", "NASDAQ"]:
        df = prepare_data(symbol)
        print(f"\n{'='*90}")
        print(f"  {symbol} - Strategy Parameter Search")
        print(f"{'='*90}")
        print(f"  {'Description':<30s} | {'AR':>7s} | {'DD':>6s} | {'SR':>6s} | {'Trades':>6s} | {'WR':>5s} | {'PF':>5s}")
        print(f"  {'-'*82}")

        for buy_t, trend, hold, sell_t, smooth, desc in configs:
            r = run_simple_backtest(df, buy_t, trend, hold, sell_t, smooth)
            marker = " ***" if r.annual_return > 0.01 else ""
            print(f"  {desc:<30s} | {r.annual_return:>6.2%} | {r.max_drawdown:>5.2%} | {r.sharpe:>6.2f} | {r.total_trades:>6d} | {r.win_rate:>4.1%} | {r.profit_factor:>5.2f}{marker}")


if __name__ == "__main__":
    main()
