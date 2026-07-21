"""Mean-reversion strategies."""

from __future__ import annotations

import pandas as pd

from src.indicators.bollinger import bollinger_bands
from src.indicators.rsi import rsi
from src.strategies.base import Strategy


class BollingerRsiMeanReversion(Strategy):
    """Enter only while price and RSI jointly breach a Bollinger Band."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_num_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        """Initialize the indicator windows and RSI entry thresholds."""
        if bb_period < 1 or rsi_period < 1:
            raise ValueError("bb_period and rsi_period must each be at least 1")
        if bb_num_std < 0:
            raise ValueError("bb_num_std must be non-negative")
        if not 0 <= rsi_oversold < rsi_overbought <= 100:
            raise ValueError(
                "RSI thresholds must satisfy 0 <= oversold < overbought <= 100"
            )

        self.bb_period = bb_period
        self.bb_num_std = bb_num_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return entry-state signals calculated only from data available then.

        Bollinger Bands and RSI use trailing windows/smoothing, so a signal at
        a row does not depend on subsequent prices.  Incomplete indicator rows
        and rows without a current joint breach remain flat.
        """
        close = data["close"]
        upper_band, _, lower_band = bollinger_bands(
            close, period=self.bb_period, num_std=self.bb_num_std
        )
        rsi_values = rsi(close, period=self.rsi_period)
        indicators_ready = (
            upper_band.notna() & lower_band.notna() & rsi_values.notna()
        )

        signals = pd.Series(0, index=data.index, dtype="int64", name="signal")
        long_entry = (close <= lower_band) & (rsi_values <= self.rsi_oversold)
        short_entry = (close >= upper_band) & (rsi_values >= self.rsi_overbought)
        signals.loc[indicators_ready & long_entry] = 1
        signals.loc[indicators_ready & short_entry] = -1
        return signals
