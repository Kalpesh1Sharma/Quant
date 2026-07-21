"""Base interface for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract interface for strategies that generate point-in-time signals."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return {-1, 0, 1} signals aligned to ``data.index``.

        Every signal at row ``t`` must use only information available at or
        before row ``t``.  Implementations must not use negative shifts,
        centered rolling windows, or statistics calculated from future rows.
        """
        raise NotImplementedError
