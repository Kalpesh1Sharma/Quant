"""Relative Strength Index indicator."""

from __future__ import annotations

import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Apply true Wilder's smoothing, correctly handling a leading NaN.

    Anchors the seed window on the first non-NaN value in ``series``
    (relevant because RSI's gains/losses come from .diff(), which starts
    with NaN). Seeds with a simple average of the first ``period`` valid
    values, then recursively smooths thereafter:
        avg_t = avg_(t-1) + (x_t - avg_(t-1)) / period

    This is NOT equivalent to ``series.ewm(alpha=1/period, adjust=False)``,
    which seeds from the very first raw data point instead.
    """
    result = pd.Series(index=series.index, dtype=float)

    valid_mask = series.notna().to_numpy()
    if not valid_mask.any():
        return result
    first_valid_pos = valid_mask.argmax()

    seed_pos = first_valid_pos + period - 1
    if seed_pos >= len(series):
        return result

    sma_seed = series.iloc[first_valid_pos : first_valid_pos + period].mean()
    result.iloc[seed_pos] = sma_seed

    for i in range(seed_pos + 1, len(series)):
        previous = result.iloc[i - 1]
        result.iloc[i] = previous + (series.iloc[i] - previous) / period

    return result


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return Wilder's Relative Strength Index on the 0--100 scale.

    Wilder's smoothing seeds with a simple moving average over the first
    ``period`` valid price-change observations, then recursively smooths
    thereafter. Because the first price change is undefined, a
    ``period``-length RSI needs ``period + 1`` prices before its first
    valid value.
    """
    if period < 1:
        raise ValueError("period must be at least 1")

    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = _wilder_smooth(gains, period)
    average_loss = _wilder_smooth(losses, period)
    relative_strength = average_gain / average_loss
    result = 100 - (100 / (1 + relative_strength))

    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    result = result.mask((average_gain == 0) & (average_loss > 0), 0.0)
    result = result.mask((average_gain == 0) & (average_loss == 0), 50.0)

    return result