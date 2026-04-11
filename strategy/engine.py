"""Strategy engine: combines specs to produce buy/sell/hold decisions with position sizing.

Key design: fixed holding period aligned with label's FORWARD_DAYS (20 trading days).
Entry when model signals high probability; exit after holding period or stop-loss.
Regime detection caps position size or blocks entries in stressed markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config import (
    PROB_BUY_THRESHOLD, STOP_LOSS_PCT, HOLD_PERIOD,
    POSITION_HIGH_CONF, POSITION_MED_CONF, REQUIRE_TREND_UP,
)
from strategy.spec import (
    MarketContext, HighProbability, TrendUp, StopLoss, and_spec,
)
from strategy.regime import detect_regime, apply_regime_cap, Regime


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Decision:
    action: Action
    position_size: float   # 0.0 to 1.0
    reason: str
    regime: str = "normal"


# Build composable buy spec
if REQUIRE_TREND_UP:
    _buy_spec = and_spec(HighProbability(PROB_BUY_THRESHOLD), TrendUp())
else:
    _buy_spec = HighProbability(PROB_BUY_THRESHOLD)

_stop_loss_spec = StopLoss(STOP_LOSS_PCT)


def _compute_position_size(ctx: MarketContext, regime: Regime) -> float:
    """Determine position size based on probability, capped by regime."""
    if ctx.prob_up > 0.8:
        raw_size = POSITION_HIGH_CONF
    elif ctx.prob_up > PROB_BUY_THRESHOLD:
        raw_size = POSITION_MED_CONF
    else:
        return 0.0
    return apply_regime_cap(raw_size, regime)


def evaluate(ctx: MarketContext) -> Decision:
    """Evaluate market context against strategy specs and return a decision."""
    regime = detect_regime(ctx.vix) if ctx.vix > 0 else Regime.NORMAL

    if ctx.in_position:
        # Stop-loss: immediate exit regardless of holding period
        if _stop_loss_spec.is_satisfied_by(ctx):
            return Decision(
                action=Action.SELL,
                position_size=0.0,
                reason=f"SELL: stop-loss (drawdown={ctx.current_drawdown:.1%})",
                regime=regime.value,
            )

        # Fixed holding period exit
        if ctx.holding_days >= HOLD_PERIOD:
            return Decision(
                action=Action.SELL,
                position_size=0.0,
                reason=f"SELL: hold complete ({ctx.holding_days}d)",
                regime=regime.value,
            )

        # Continue holding
        size = _compute_position_size(ctx, regime)
        return Decision(
            action=Action.HOLD,
            position_size=size,
            reason=f"HOLD: day {ctx.holding_days}/{HOLD_PERIOD}",
            regime=regime.value,
        )

    # STRESS regime: no new entries
    if regime == Regime.STRESS:
        return Decision(
            action=Action.HOLD,
            position_size=0.0,
            reason=f"HOLD: regime=STRESS (VIX={ctx.vix:.0f}), no new entries",
            regime=regime.value,
        )

    # Check buy signal
    if _buy_spec.is_satisfied_by(ctx):
        size = _compute_position_size(ctx, regime)
        if size > 0:
            regime_note = f", regime={regime.value}" if regime != Regime.NORMAL else ""
            return Decision(
                action=Action.BUY,
                position_size=size,
                reason=f"BUY: prob={ctx.prob_up:.2f}, pos={size:.0%}{regime_note}",
                regime=regime.value,
            )

    return Decision(
        action=Action.HOLD,
        position_size=0.0,
        reason=f"HOLD: prob={ctx.prob_up:.2f}, waiting",
        regime=regime.value,
    )
