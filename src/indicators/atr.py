"""Average True Range indicator."""

from __future__ import annotations

import pandas as pd

from src.indicators.rsi import _wilder_smooth


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Return Wilder's Average True Range.

    True range is the largest of high-low, high-previous-close, and
    low-previous-close. The first true range uses high-low because no
    prior close exists. Wilder's smoothing seeds with a simple moving
    average over the first ``period`` true range values, then recursively
    smooths thereafter; the first ``period - 1`` ATR values are ``NaN``.
    """
    if period < 1:
        raise ValueError("period must be at least 1")
    if not high.index.equals(low.index) or not high.index.equals(close.index):
        raise ValueError("high, low, and close must have identical indexes")

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _wilder_smooth(true_range, period)