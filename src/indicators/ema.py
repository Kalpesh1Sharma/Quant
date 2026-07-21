"""Exponential moving average indicator."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Return the trading-style exponential moving average of ``series``.

    The first ``span - 1`` values are ``NaN`` so callers can distinguish the
    warm-up period from values that are valid for trading decisions.
    """
    if span < 1:
        raise ValueError("span must be at least 1")

    return series.ewm(span=span, adjust=False, min_periods=span).mean()
