from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_features
from src.indicators.atr import atr
from src.indicators.bollinger import bollinger_bands
from src.indicators.ema import ema
from src.indicators.macd import macd
from src.indicators.rsi import rsi


def _ohlcv(length: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC")
    close = pd.Series(100.0 + np.arange(length), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1_000.0 + (10.0 * np.arange(length)),
        },
        index=index,
    )


def test_feature_groups_match_expected_indicator_and_rolling_values() -> None:
    data = _ohlcv()
    features = build_features(data)
    row = 35
    close = data["close"]
    daily_return = close.pct_change(fill_method=None)
    macd_line, macd_signal, macd_histogram = macd(close)
    upper_band, _, lower_band = bollinger_bands(close)

    assert features.iloc[row]["return_1d"] == pytest.approx(daily_return.iloc[row])
    assert features.iloc[row]["return_5d"] == pytest.approx(close.pct_change(5).iloc[row])
    assert features.iloc[row]["return_10d"] == pytest.approx(close.pct_change(10).iloc[row])

    assert features.iloc[row]["ema_12_rel"] == pytest.approx(
        (ema(close, 12).iloc[row] / close.iloc[row]) - 1
    )
    assert features.iloc[row]["ema_26_rel"] == pytest.approx(
        (ema(close, 26).iloc[row] / close.iloc[row]) - 1
    )
    assert features.iloc[row]["rsi_14"] == pytest.approx(rsi(close, 14).iloc[row])
    assert features.iloc[row]["macd_line_rel"] == pytest.approx(
        macd_line.iloc[row] / close.iloc[row]
    )
    assert features.iloc[row]["macd_signal_rel"] == pytest.approx(
        macd_signal.iloc[row] / close.iloc[row]
    )
    assert features.iloc[row]["macd_histogram_rel"] == pytest.approx(
        macd_histogram.iloc[row] / close.iloc[row]
    )
    assert features.iloc[row]["bollinger_percent_b"] == pytest.approx(
        (close.iloc[row] - lower_band.iloc[row])
        / (upper_band.iloc[row] - lower_band.iloc[row])
    )
    assert features.iloc[row]["atr_14_rel"] == pytest.approx(
        atr(data["high"], data["low"], close, 14).iloc[row] / close.iloc[row]
    )

    assert features.iloc[row]["realized_volatility_10d"] == pytest.approx(
        daily_return.rolling(10, min_periods=10).std().iloc[row]
    )
    assert features.iloc[row]["realized_volatility_20d"] == pytest.approx(
        daily_return.rolling(20, min_periods=20).std().iloc[row]
    )
    expected_volume_zscore = (
        (data["volume"] - data["volume"].rolling(10, min_periods=10).mean())
        / data["volume"].rolling(10, min_periods=10).std()
    )
    assert features.iloc[row]["volume_zscore_10d"] == pytest.approx(
        expected_volume_zscore.iloc[row]
    )

    assert features.iloc[row]["return_1d_lag_1"] == pytest.approx(
        daily_return.iloc[row - 1]
    )
    assert features.iloc[row]["return_1d_lag_2"] == pytest.approx(
        daily_return.iloc[row - 2]
    )
    assert features.iloc[row]["return_1d_lag_3"] == pytest.approx(
        daily_return.iloc[row - 3]
    )


def test_features_never_include_raw_ohlc_columns() -> None:
    features = build_features(_ohlcv())

    assert {"open", "high", "low", "close"}.isdisjoint(features.columns)
