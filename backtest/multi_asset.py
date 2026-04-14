"""Multi-asset portfolio backtesting engine.

Combines signals from multiple assets (SP500 + NASDAQ) into a single
portfolio, with per-asset position caps. Diversification improves Sharpe
by smoothing the equity curve across uncorrelated signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from config import (
    RISK_FREE_RATE, PER_ASSET_MAX_POSITION, MAX_TOTAL_EXPOSURE,
    PROB_BUY_THRESHOLD, STOP_LOSS_PCT, HOLD_PERIOD,
    POSITION_HIGH_CONF, POSITION_MED_CONF, COST_ROUND_TRIP,
)
from backtest.cost_model import apply_buy_cost, apply_sell_cost
from backtest.engine import Trade, BacktestResult
from backtest.signals import collect_asset_signals
from strategy.regime import detect_regime, apply_regime_cap, Regime


def run_multi_asset_backtest(
    prepared_dfs: dict[str, pd.DataFrame],
) -> BacktestResult:
    """Run multi-asset walk-forward backtest.

    Args:
        prepared_dfs: dict mapping symbol keys to prepared DataFrames
                      (with features and labels already added)

    Returns:
        BacktestResult with combined portfolio metrics
    """
    # Collect signals for each asset
    asset_signals = {
        symbol: collect_asset_signals(df, symbol)
        for symbol, df in prepared_dfs.items()
    }

    # Union of all dates
    all_dates = sorted(set().union(*(s.keys() for s in asset_signals.values())))
    if not all_dates:
        return BacktestResult(symbol="MULTI")

    asset_names = "+".join(sorted(prepared_dfs.keys()))
    result = BacktestResult(symbol=f"MULTI({asset_names})")

    portfolio = 1.0
    positions = {}  # symbol -> {entry_price, entry_idx, size, entry_date}
    first_closes = {}

    for i, date_str in enumerate(all_dates):
        day_pnl = 0.0

        # Check exits for all positions
        to_close = []
        for sym, pos in positions.items():
            if date_str not in asset_signals.get(sym, {}):
                continue
            d = asset_signals[sym][date_str]
            close = d["close"]
            days_held = i - pos["entry_idx"]
            dd = max(0, (pos["entry_price"] - close) / pos["entry_price"])

            if days_held >= HOLD_PERIOD or dd > STOP_LOSS_PCT:
                exit_price = apply_sell_cost(close)
                net_ret = exit_price / pos["entry_price"] - 1
                trade = Trade(
                    entry_date=pos["entry_date"],
                    exit_date=date_str,
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    position_size=pos["size"],
                    gross_return=net_ret + COST_ROUND_TRIP,
                    net_return=net_ret,
                    holding_days=days_held,
                )
                result.trades.append(trade)
                day_pnl += net_ret * pos["size"]
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        # Check entries (prioritize highest probability)
        candidates = []
        for sym, signals in asset_signals.items():
            if sym in positions or date_str not in signals:
                continue
            d = signals[date_str]
            regime = detect_regime(d["vix"]) if d["vix"] > 0 else Regime.NORMAL
            if regime == Regime.STRESS:
                continue
            if d["prob"] > PROB_BUY_THRESHOLD:
                candidates.append((d["prob"], sym, d, regime))

        candidates.sort(reverse=True)  # highest probability first
        current_exposure = sum(p["size"] for p in positions.values())
        for prob, sym, d, regime in candidates:
            raw_size = POSITION_HIGH_CONF if prob > 0.8 else POSITION_MED_CONF
            size = apply_regime_cap(raw_size, regime)
            size = min(size, PER_ASSET_MAX_POSITION)
            # Cap total portfolio exposure
            if current_exposure + size > MAX_TOTAL_EXPOSURE:
                size = MAX_TOTAL_EXPOSURE - current_exposure
                if size <= 0.05:
                    continue  # skip if remaining capacity too small
            entry_price = apply_buy_cost(d["close"])
            positions[sym] = {
                "entry_price": entry_price,
                "entry_idx": i,
                "size": size,
                "entry_date": date_str,
            }
            current_exposure += size

        portfolio *= (1 + day_pnl)
        result.portfolio_values.append(portfolio)
        result.portfolio_dates.append(date_str)

        # Buy & hold: average of all assets
        bh_vals = []
        for sym, signals in asset_signals.items():
            if date_str in signals:
                close = signals[date_str]["close"]
                if sym not in first_closes:
                    first_closes[sym] = close
                bh_vals.append(close / first_closes[sym])
        result.buy_hold_values.append(np.mean(bh_vals) if bh_vals else 1.0)

    # Close remaining positions
    for sym, pos in positions.items():
        last_date = all_dates[-1]
        if last_date in asset_signals.get(sym, {}):
            d = asset_signals[sym][last_date]
            exit_price = apply_sell_cost(d["close"])
            net_ret = exit_price / pos["entry_price"] - 1
            days_held = len(all_dates) - 1 - pos["entry_idx"]
            trade = Trade(
                entry_date=pos["entry_date"],
                exit_date=last_date,
                entry_price=pos["entry_price"],
                exit_price=exit_price,
                position_size=pos["size"],
                gross_return=net_ret + COST_ROUND_TRIP,
                net_return=net_ret,
                holding_days=days_held,
            )
            result.trades.append(trade)
            portfolio *= (1 + net_ret * pos["size"])
    if positions:
        result.portfolio_values[-1] = portfolio

    return result
