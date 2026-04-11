import pandas as pd

from config import FORWARD_DAYS, LABEL_THRESHOLD


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary label: 1 if 5-day forward return > LABEL_THRESHOLD (1%), else 0.

    Uses T+1 to T+N data strictly (shift by -FORWARD_DAYS on close).
    Rows without valid future data get NaN label.
    """
    df = df.copy()
    future_return = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
    df["future_return"] = future_return
    df["label"] = (future_return > LABEL_THRESHOLD).astype(float)
    df.loc[df["future_return"].isna(), "label"] = float("nan")
    return df
