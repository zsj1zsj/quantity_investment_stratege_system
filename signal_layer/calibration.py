import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

from config import SIGNAL_SMOOTH_SPAN


def calibrate_model(model: BaseEstimator, X_cal: np.ndarray, y_cal: np.ndarray) -> CalibratedClassifierCV:
    """Calibrate model probabilities using Platt Scaling (sigmoid).

    Uses a held-out calibration set to fit the calibrator.
    Returns a CalibratedClassifierCV wrapper that outputs calibrated probabilities.
    """
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def smooth_signal(probs: pd.Series, span: int = SIGNAL_SMOOTH_SPAN) -> pd.Series:
    """Apply exponential weighted moving average to smooth signal noise.

    Args:
        probs: Series of raw probability values (indexed by date).
        span: EWM span in days (default 3).

    Returns:
        Smoothed signal values in [0, 1].
    """
    smoothed = probs.ewm(span=span, adjust=False).mean()
    return smoothed.clip(0, 1)


def process_signal(raw_probs: pd.Series, apply_smoothing: bool = True) -> pd.Series:
    """Full signal processing pipeline: optional smoothing.

    Calibration is applied at model level (CalibratedClassifierCV wraps the model).
    This function handles post-calibration signal processing.
    """
    signal = raw_probs.copy()
    if apply_smoothing and len(signal) > 1:
        signal = smooth_signal(signal)
    return signal
