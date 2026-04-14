"""Market regime detection based on VIX levels.

Regime states:
  - NORMAL:  VIX < 18  — full position allowed
  - CAUTION: 18 <= VIX < 25 — position cap at 50%
  - STRESS:  VIX >= 25 — no new entries, only stop-loss exits
"""
from enum import Enum


class Regime(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    STRESS = "stress"


# VIX thresholds (tightened in Phase 2.5 for better drawdown control)
VIX_CAUTION = 18
VIX_STRESS = 25

# Position cap per regime
REGIME_POSITION_CAP = {
    Regime.NORMAL: 1.0,    # no cap
    Regime.CAUTION: 0.5,   # max 50%
    Regime.STRESS: 0.0,    # no new entries
}


def detect_regime(vix: float) -> Regime:
    """Classify market regime based on VIX value using default thresholds."""
    return detect_regime_custom(vix, VIX_CAUTION, VIX_STRESS)


def detect_regime_custom(vix: float, caution: float, stress: float) -> Regime:
    """Classify market regime with explicit thresholds (used by sensitivity tests)."""
    if vix >= stress:
        return Regime.STRESS
    if vix >= caution:
        return Regime.CAUTION
    return Regime.NORMAL


def apply_regime_cap(position_size: float, regime: Regime) -> float:
    """Apply regime-based position cap."""
    cap = REGIME_POSITION_CAP[regime]
    return min(position_size, cap)
