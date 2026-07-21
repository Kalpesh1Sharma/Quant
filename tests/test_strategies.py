from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion import BollingerRsiMeanReversion
from src.strategies.momentum import EmaCrossoverMomentum


def _market_data(prices: list[float]) -> pd.DataFrame:
    """Build the OHLCV shape consumed by strategies from closing prices."""
    index = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    close = pd.Series(prices, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
        },
        index=index,
    )


@pytest.fixture
def price_data() -> pd.DataFrame:
    prices = (100 + np.linspace(-4, 4, 30) + 3 * np.sin(np.arange(30))).tolist()
    return _market_data(prices)


def test_momentum_signals_are_invariant_to_future_data(
    price_data: pd.DataFrame,
) -> None:
    strategy = EmaCrossoverMomentum(fast_span=3, slow_span=6)

    full_signals = strategy.generate_signals(price_data)
    prefix_signals = strategy.generate_signals(price_data.iloc[:20])

    pd.testing.assert_series_equal(full_signals.iloc[:20], prefix_signals)


def test_mean_reversion_signals_are_invariant_to_future_data(
    price_data: pd.DataFrame,
) -> None:
    strategy = BollingerRsiMeanReversion(bb_period=5, rsi_period=4)

    full_signals = strategy.generate_signals(price_data)
    prefix_signals = strategy.generate_signals(price_data.iloc[:20])

    pd.testing.assert_series_equal(full_signals.iloc[:20], prefix_signals)


@pytest.mark.parametrize(
    "strategy",
    [
        EmaCrossoverMomentum(fast_span=3, slow_span=6),
        BollingerRsiMeanReversion(bb_period=5, rsi_period=4),
    ],
)
def test_signal_values_are_only_negative_one_zero_or_one(
    strategy, price_data: pd.DataFrame
) -> None:
    signals = strategy.generate_signals(price_data)

    assert signals.isin([-1, 0, 1]).all()


def test_momentum_signal_is_flat_during_indicator_warmup(
    price_data: pd.DataFrame,
) -> None:
    strategy = EmaCrossoverMomentum(fast_span=3, slow_span=6)

    signals = strategy.generate_signals(price_data)

    assert (signals.iloc[:5] == 0).all()


def test_mean_reversion_signal_is_flat_during_indicator_warmup(
    price_data: pd.DataFrame,
) -> None:
    strategy = BollingerRsiMeanReversion(bb_period=5, rsi_period=4)

    signals = strategy.generate_signals(price_data)

    assert (signals.iloc[:4] == 0).all()


@pytest.mark.parametrize(
    "strategy",
    [
        EmaCrossoverMomentum(fast_span=3, slow_span=6),
        BollingerRsiMeanReversion(bb_period=5, rsi_period=4),
    ],
)
def test_signal_index_matches_input_index(strategy, price_data: pd.DataFrame) -> None:
    signals = strategy.generate_signals(price_data)

    assert signals.index.equals(price_data.index)


def test_momentum_uses_the_crossover_state_after_each_cross() -> None:
    data = _market_data([10, 9, 8, 7, 10, 12, 14, 6, 4])
    strategy = EmaCrossoverMomentum(fast_span=2, slow_span=3)

    signals = strategy.generate_signals(data)

    expected = pd.Series([0, 0, -1, -1, 1, 1, 1, -1, -1], index=data.index)
    pd.testing.assert_series_equal(signals, expected, check_names=False)


def test_mean_reversion_fires_on_breach_and_resets_inside_bands() -> None:
    data = _market_data([100, 100, 100, 90, 100])
    strategy = BollingerRsiMeanReversion(
        bb_period=3, bb_num_std=1.0, rsi_period=2, rsi_oversold=30.0
    )

    signals = strategy.generate_signals(data)

    assert signals.iloc[3] == 1
    assert signals.iloc[4] == 0
