"""Momentum strategies."""

from __future__ import annotations

import pandas as pd

from src.indicators.ema import ema
from src.strategies.base import Strategy


class EmaCrossoverMomentum(Strategy):
    """Hold the current direction implied by a pair of exponential averages."""

    def __init__(self, fast_span: int = 12, slow_span: int = 26) -> None:
        """Initialize the strategy with its fast and slow EMA spans."""
        if fast_span < 1 or slow_span < 1:
            raise ValueError("fast_span and slow_span must each be at least 1")
        if fast_span >= slow_span:
            raise ValueError("fast_span must be smaller than slow_span")

        self.fast_span = fast_span
        self.slow_span = slow_span

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return the current EMA-crossover state without using future rows.

        Each row uses only its current and prior closing prices through the
        trailing EMAs.  Rows before both EMAs are available remain flat.
        """
        close = data["close"]
        fast_ema = ema(close, self.fast_span)
        slow_ema = ema(close, self.slow_span)

        signals = pd.Series(0, index=data.index, dtype="int64", name="signal")
        signals.loc[(fast_ema > slow_ema) & slow_ema.notna()] = 1
        signals.loc[(fast_ema < slow_ema) & slow_ema.notna()] = -1
        return signals
