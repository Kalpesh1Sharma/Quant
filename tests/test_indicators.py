from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.atr import atr
from src.indicators.bollinger import bollinger_bands
from src.indicators.ema import ema
from src.indicators.macd import macd
from src.indicators.rsi import rsi


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series(
        [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0, 17.0],
        index=pd.date_range("2024-01-01", periods=10, freq="D", name="date"),
        name="close",
    )


def test_ema_uses_trading_convention_and_preserves_warm_up(
    prices: pd.Series,
) -> None:
    result = ema(prices, span=3)
    expected = prices.ewm(span=3, adjust=False, min_periods=3).mean()

    pd.testing.assert_series_equal(result, expected)
    assert result.index.equals(prices.index)
    assert result.iloc[:2].isna().all()


def test_rsi_matches_wilder_reference_calculation() -> None:
    # Classic 14-period RSI worked-example price series. A 14-period RSI
    # needs 14 price changes, i.e. 15 prices, so the first valid value
    # lands at index 14 (0-based) -- the 15th data point, using deltas
    # from indices 1 through 14.
    prices = pd.Series([
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
        45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
    ])
    result = rsi(prices, period=14)

    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(70.5, abs=1.0)


def test_rsi_preserves_warm_up_length(prices: pd.Series) -> None:
    result = rsi(prices, period=3)

    assert result.index.equals(prices.index)
    assert result.iloc[:3].isna().all()
    assert pd.notna(result.iloc[3])


def test_rsi_returns_conventional_endpoints_for_one_sided_or_flat_prices() -> None:
    rising = pd.Series([1.0, 2.0, 3.0, 4.0])
    falling = pd.Series([4.0, 3.0, 2.0, 1.0])
    flat = pd.Series([2.0, 2.0, 2.0, 2.0])

    assert rsi(rising, period=2).iloc[-1] == 100.0
    assert rsi(falling, period=2).iloc[-1] == 0.0
    assert rsi(flat, period=2).iloc[-1] == 50.0


def test_macd_uses_ema_components_and_delays_signal_warm_up(
    prices: pd.Series,
) -> None:
    macd_line, signal_line, histogram = macd(prices, fast=2, slow=4, signal=3)
    expected_macd = ema(prices, 2) - ema(prices, 4)
    expected_signal = ema(expected_macd, 3)

    pd.testing.assert_series_equal(macd_line, expected_macd)
    pd.testing.assert_series_equal(signal_line, expected_signal)
    pd.testing.assert_series_equal(histogram, expected_macd - expected_signal)
    assert macd_line.iloc[:3].isna().all()
    assert signal_line.iloc[:5].isna().all()
    assert all(result.index.equals(prices.index) for result in (macd_line, signal_line, histogram))


def test_bollinger_bands_use_population_standard_deviation(
    prices: pd.Series,
) -> None:
    upper, middle, lower = bollinger_bands(prices, period=3, num_std=2.0)
    expected_middle = prices.rolling(3, min_periods=3).mean()
    expected_std = prices.rolling(3, min_periods=3).std(ddof=0)

    pd.testing.assert_series_equal(middle, expected_middle)
    pd.testing.assert_series_equal(upper, expected_middle + (2 * expected_std))
    pd.testing.assert_series_equal(lower, expected_middle - (2 * expected_std))
    assert all(result.iloc[:2].isna().all() for result in (upper, middle, lower))


def test_atr_matches_wilder_seeded_calculation() -> None:
    # True ranges here are: [2.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    # Wilder seed at index 2 (period=3) = mean(2.0, 3.0, 3.0) = 2.6667
    # index 3 = 2.6667 + (4.0 - 2.6667) / 3 = 3.1111
    # index 4 = 3.1111 + (4.0 - 3.1111) / 3 = 3.4074
    # Computed independently from Wilder's SMA-seeded recursive
    # definition — not re-derived from this implementation's own
    # ewm-based formula.
    index = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    high = pd.Series([10.0, 12.0, 13.0, 15.0, 14.0, 16.0], index=index)
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 10.0, 12.0], index=index)
    close = pd.Series([9.0, 11.0, 11.0, 12.0, 11.0, 15.0], index=index)

    result = atr(high, low, close, period=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.6667, abs=0.001)
    assert result.iloc[3] == pytest.approx(3.1111, abs=0.001)
    assert result.iloc[4] == pytest.approx(3.4074, abs=0.001)
    assert result.index.equals(index)


def test_atr_rejects_misaligned_series() -> None:
    index = pd.RangeIndex(3)
    with pytest.raises(ValueError, match="identical indexes"):
        atr(
            pd.Series([3.0, 4.0, 5.0], index=index),
            pd.Series([1.0, 2.0, 3.0], index=index),
            pd.Series([2.0, 3.0, 4.0], index=pd.RangeIndex(1, 4)),
        )


@pytest.mark.parametrize("function, arguments", [
    (ema, (pd.Series([1.0]), 0)),
    (rsi, (pd.Series([1.0]), 0)),
    (bollinger_bands, (pd.Series([1.0]), 0)),
])
def test_period_parameters_must_be_positive(function, arguments) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        function(*arguments)


def test_indicator_values_are_finite_after_warm_up(prices: pd.Series) -> None:
    values = [
        ema(prices, 3).iloc[2:],
        rsi(prices, 3).iloc[3:],
        *[band.iloc[2:] for band in bollinger_bands(prices, 3)],
    ]

    assert all(np.isfinite(value).all() for value in values)