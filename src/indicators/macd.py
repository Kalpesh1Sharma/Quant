"""Moving Average Convergence Divergence indicator."""

from __future__ import annotations

import pandas as pd

from src.indicators.ema import ema


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return the MACD line, signal line, and histogram.

    All component moving averages retain their natural warm-up period as
    ``NaN``.  ``slow`` must exceed ``fast`` for the conventional MACD setup.
    """
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("fast, slow, and signal must each be at least 1")
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")

    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
