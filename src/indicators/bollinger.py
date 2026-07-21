"""Bollinger Bands indicator."""

from __future__ import annotations

import pandas as pd


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return upper, middle, and lower Bollinger Bands.

    The bands use the population standard deviation (``ddof=0``), which
    treats every value in the rolling window as the full population.  The
    first ``period - 1`` values of each band are ``NaN``.
    """
    if period < 1:
        raise ValueError("period must be at least 1")
    if num_std < 0:
        raise ValueError("num_std must be non-negative")

    rolling = series.rolling(window=period, min_periods=period)
    middle_band = rolling.mean()
    standard_deviation = rolling.std(ddof=0)
    upper_band = middle_band + (num_std * standard_deviation)
    lower_band = middle_band - (num_std * standard_deviation)
    return upper_band, middle_band, lower_band
