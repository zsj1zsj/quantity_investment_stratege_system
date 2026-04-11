"""Specification Pattern for composable trading rules.

Each spec answers: "Given the current market context, should this condition be met?"
Specs can be combined with and_spec / or_spec / not_spec for transparent, testable logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class MarketContext:
    """Snapshot of market state at a given point in time."""
    date: str
    close: float
    prob_up: float              # calibrated signal
    ma5: float
    ma20: float
    close_ma20_ratio: float     # close / MA20
    rsi: float
    volatility_5d: float
    volatility_20d: float
    vol_ratio: float            # volatility_5d / volatility_20d
    macd_hist: float
    vix: float = 0.0               # VIX value for regime detection
    # Position tracking
    in_position: bool = False
    entry_price: float = 0.0
    holding_days: int = 0
    current_drawdown: float = 0.0  # unrealized loss from entry


class TradingSpec(Protocol):
    """Protocol for trading specifications."""
    def is_satisfied_by(self, ctx: MarketContext) -> bool: ...


class HighProbability:
    """Signal probability exceeds threshold."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return ctx.prob_up > self.threshold

    def __repr__(self) -> str:
        return f"HighProbability(>{self.threshold})"


class LowProbability:
    """Signal probability below threshold."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return ctx.prob_up < self.threshold

    def __repr__(self) -> str:
        return f"LowProbability(<{self.threshold})"


class TrendUp:
    """Price is above MA20 (uptrend)."""
    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return ctx.close_ma20_ratio > 1.0

    def __repr__(self) -> str:
        return "TrendUp(close>MA20)"


class VolatilityNormal:
    """Volatility ratio within normal range (not extreme)."""
    def __init__(self, max_ratio: float = 1.5):
        self.max_ratio = max_ratio

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return ctx.vol_ratio <= self.max_ratio

    def __repr__(self) -> str:
        return f"VolatilityNormal(ratio<={self.max_ratio})"


class StopLoss:
    """Current drawdown exceeds stop-loss threshold."""
    def __init__(self, max_drawdown: float = 0.08):
        self.max_drawdown = max_drawdown

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return ctx.in_position and ctx.current_drawdown >= self.max_drawdown

    def __repr__(self) -> str:
        return f"StopLoss(dd>={self.max_drawdown:.0%})"


class AndSpec:
    """Logical AND of two specs."""
    def __init__(self, left: TradingSpec, right: TradingSpec):
        self.left = left
        self.right = right

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return self.left.is_satisfied_by(ctx) and self.right.is_satisfied_by(ctx)

    def __repr__(self) -> str:
        return f"({self.left} AND {self.right})"


class OrSpec:
    """Logical OR of two specs."""
    def __init__(self, left: TradingSpec, right: TradingSpec):
        self.left = left
        self.right = right

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return self.left.is_satisfied_by(ctx) or self.right.is_satisfied_by(ctx)

    def __repr__(self) -> str:
        return f"({self.left} OR {self.right})"


class NotSpec:
    """Logical NOT of a spec."""
    def __init__(self, spec: TradingSpec):
        self.spec = spec

    def is_satisfied_by(self, ctx: MarketContext) -> bool:
        return not self.spec.is_satisfied_by(ctx)

    def __repr__(self) -> str:
        return f"NOT({self.spec})"


def and_spec(left: TradingSpec, right: TradingSpec) -> AndSpec:
    return AndSpec(left, right)


def or_spec(left: TradingSpec, right: TradingSpec) -> OrSpec:
    return OrSpec(left, right)


def not_spec(spec: TradingSpec) -> NotSpec:
    return NotSpec(spec)
