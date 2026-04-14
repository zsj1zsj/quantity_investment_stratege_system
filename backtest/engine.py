"""Walk-forward backtesting engine with transaction costs.

Trains model on each sliding window, generates raw signals on test window,
runs strategy decisions (fixed holding period), and tracks portfolio performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from dataclasses import dataclass, field

from config import (
    LGBM_PARAMS, TRAIN_WINDOW, TEST_WINDOW, STEP_SIZE,
    RISK_FREE_RATE, GRADUAL_ENTRY, GRADUAL_INITIAL_SIZE,
    GRADUAL_ADD_DAY, GRADUAL_ADD_THRESHOLD, COST_ROUND_TRIP,
)
from model.train import _get_feature_cols
from strategy.spec import MarketContext
from strategy.engine import evaluate, Action
from backtest.cost_model import apply_buy_cost, apply_sell_cost


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position_size: float
    gross_return: float
    net_return: float
    holding_days: int


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    portfolio_values: list[float] = field(default_factory=list)
    portfolio_dates: list[str] = field(default_factory=list)
    buy_hold_values: list[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.net_return > 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        wins = [t.net_return for t in self.trades if t.net_return > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_return for t in self.trades if t.net_return <= 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.net_return for t in self.trades if t.net_return > 0)
        gross_loss = abs(sum(t.net_return for t in self.trades if t.net_return <= 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def net_annual_return(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0.0
        total_return = self.portfolio_values[-1] / self.portfolio_values[0]
        n_days = len(self.portfolio_values)
        years = n_days / 252
        if years <= 0:
            return 0.0
        return total_return ** (1 / years) - 1

    def buy_hold_annual_return(self) -> float:
        if len(self.buy_hold_values) < 2:
            return 0.0
        total_return = self.buy_hold_values[-1] / self.buy_hold_values[0]
        n_days = len(self.buy_hold_values)
        years = n_days / 252
        if years <= 0:
            return 0.0
        return total_return ** (1 / years) - 1

    def max_drawdown(self) -> float:
        if not self.portfolio_values:
            return 0.0
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        return float(np.max(drawdowns))

    def sharpe_ratio(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0.0
        values = np.array(self.portfolio_values)
        daily_returns = np.diff(values) / values[:-1]
        if np.std(daily_returns) == 0:
            return 0.0
        annual_return = self.net_annual_return()
        annual_vol = np.std(daily_returns) * np.sqrt(252)
        return (annual_return - RISK_FREE_RATE) / annual_vol


def run_backtest(df: pd.DataFrame, symbol: str) -> BacktestResult:
    """Run walk-forward backtest on prepared data.

    Uses raw model probabilities (no calibration/smoothing) to avoid
    compressing the signal distribution. Strategy uses fixed holding period
    aligned with the label's FORWARD_DAYS.
    """
    feature_cols = _get_feature_cols()
    result = BacktestResult(symbol=symbol)

    n = len(df)

    # Step 1: Walk-forward to collect all out-of-sample predictions
    all_test_data = []

    start = 0
    while start + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_end = start + TRAIN_WINDOW
        test_end = min(train_end + TEST_WINDOW, n)

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        # Use raw probabilities - no calibration or smoothing
        probs = model.predict_proba(X_test)[:, 1]

        for idx, row_idx in enumerate(test_df.index):
            row = test_df.loc[row_idx]
            all_test_data.append({
                "date": row_idx,
                "close": row["close"],
                "prob_up": probs[idx],
                "ma5": row.get("ma5", 0),
                "ma20": row.get("ma20", 0),
                "close_ma20_ratio": row.get("close_ma20_ratio", 1),
                "rsi": row.get("rsi", 50),
                "volatility_5d": row.get("volatility_5d", 0),
                "volatility_20d": row.get("volatility_20d", 0),
                "vol_ratio": row.get("vol_ratio", 1),
                "macd_hist": row.get("macd_hist", 0),
                "vix": row.get("vix", 0),
            })

        start += STEP_SIZE

    # Deduplicate overlapping windows - keep first occurrence
    seen_dates = set()
    unique_data = []
    for d in all_test_data:
        date_key = str(d["date"])
        if date_key not in seen_dates:
            seen_dates.add(date_key)
            unique_data.append(d)

    unique_data.sort(key=lambda x: x["date"])

    if not unique_data:
        return result

    # Step 2: Simulate trading sequentially
    portfolio_value = 1.0
    in_position = False
    entry_price = 0.0
    entry_date = ""
    position_size = 0.0
    holding_days = 0
    first_close = unique_data[0]["close"]

    for data in unique_data:
        close = data["close"]
        current_drawdown = 0.0
        if in_position and entry_price > 0:
            current_drawdown = max(0, (entry_price - close) / entry_price)
            holding_days += 1

        ctx = MarketContext(
            date=str(data["date"]),
            close=close,
            prob_up=data["prob_up"],
            ma5=data["ma5"],
            ma20=data["ma20"],
            close_ma20_ratio=data["close_ma20_ratio"],
            rsi=data["rsi"],
            volatility_5d=data["volatility_5d"],
            volatility_20d=data["volatility_20d"],
            vol_ratio=data["vol_ratio"],
            macd_hist=data["macd_hist"],
            vix=data.get("vix", 0),
            in_position=in_position,
            entry_price=entry_price,
            holding_days=holding_days,
            current_drawdown=current_drawdown,
        )

        decision = evaluate(ctx)

        if decision.action == Action.BUY and not in_position:
            entry_price = apply_buy_cost(close)
            entry_date = str(data["date"])
            target_size = decision.position_size
            # Gradual entry: start with initial size, add later
            if GRADUAL_ENTRY:
                position_size = min(GRADUAL_INITIAL_SIZE, target_size)
            else:
                position_size = target_size
            in_position = True
            holding_days = 0

        elif decision.action == Action.HOLD and in_position:
            # Gradual position building: add on day GRADUAL_ADD_DAY if signal still valid
            if (GRADUAL_ENTRY
                and holding_days == GRADUAL_ADD_DAY
                and data["prob_up"] > GRADUAL_ADD_THRESHOLD
                and position_size < decision.position_size):
                position_size = decision.position_size

        elif decision.action == Action.SELL and in_position:
            exit_price = apply_sell_cost(close)
            net_ret = (exit_price / entry_price) - 1
            trade = Trade(
                entry_date=entry_date,
                exit_date=str(data["date"]),
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                gross_return=net_ret + COST_ROUND_TRIP,  # net + round-trip cost
                net_return=net_ret,
                holding_days=holding_days,
            )
            result.trades.append(trade)

            portfolio_value *= (1 + net_ret * position_size)
            in_position = False
            entry_price = 0.0
            position_size = 0.0
            holding_days = 0

        result.portfolio_values.append(portfolio_value)
        result.portfolio_dates.append(str(data["date"]))
        result.buy_hold_values.append(close / first_close)

    # Close any open position at end
    if in_position and unique_data:
        last_close = unique_data[-1]["close"]
        exit_price = apply_sell_cost(last_close)
        net_ret = (exit_price / entry_price) - 1
        trade = Trade(
            entry_date=entry_date,
            exit_date=str(unique_data[-1]["date"]),
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            gross_return=net_ret + COST_ROUND_TRIP,
            net_return=net_ret,
            holding_days=holding_days,
        )
        result.trades.append(trade)
        portfolio_value *= (1 + net_ret * position_size)
        result.portfolio_values[-1] = portfolio_value

    return result


def print_backtest_report(result: BacktestResult) -> None:
    """Print formatted backtest results."""
    print(f"\n{'='*60}")
    print(f"  Backtest Report: {result.symbol}")
    print(f"{'='*60}")

    if not result.trades:
        print("  No trades executed.")
        return

    period = f"{result.portfolio_dates[0]} ~ {result.portfolio_dates[-1]}"
    print(f"  Period: {period}")
    print(f"  Trading days: {len(result.portfolio_values)}")
    print()

    # Strategy performance
    print("  Strategy Performance (net of costs):")
    print(f"    Annual Return:  {result.net_annual_return():>8.2%}")
    print(f"    Max Drawdown:   {result.max_drawdown():>8.2%}")
    print(f"    Sharpe Ratio:   {result.sharpe_ratio():>8.2f}")
    print()

    # Buy & Hold baseline
    print("  Buy & Hold Baseline:")
    print(f"    Annual Return:  {result.buy_hold_annual_return():>8.2%}")
    print()

    # Trade statistics
    print("  Trade Statistics:")
    print(f"    Total Trades:   {result.total_trades:>8d}")
    years = len(result.portfolio_values) / 252
    if years > 0:
        print(f"    Trades/Year:    {result.total_trades / years:>8.1f}")
    print(f"    Win Rate:       {result.win_rate:>8.2%}")
    print(f"    Avg Win:        {result.avg_win:>8.2%}")
    print(f"    Avg Loss:       {result.avg_loss:>8.2%}")
    print(f"    Profit Factor:  {result.profit_factor:>8.2f}")
    print()

    # Gate 1 checks
    print("  Gate 1 Checks:")
    annual_ret = result.net_annual_return()
    max_dd = result.max_drawdown()
    print(f"    Net annual return > 0%:   {'PASS' if annual_ret > 0 else 'FAIL'} ({annual_ret:.2%})")
    print(f"    Max drawdown < 30%:       {'PASS' if max_dd < 0.30 else 'FAIL'} ({max_dd:.2%})")
    print(f"{'='*60}")
