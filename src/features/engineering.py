"""Leakage-safe feature and supervised-label construction for OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.atr import atr
from src.indicators.bollinger import bollinger_bands
from src.indicators.ema import ema
from src.indicators.macd import macd
from src.indicators.rsi import rsi


_REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_LABEL_COLUMN_NAMES = {"label", "labels", "target", "targets", "forward_label"}


def _validate_ohlcv_input(data: pd.DataFrame) -> None:
    """Validate raw OHLCV input and reject a target mixed into its features."""
    label_columns = [
        str(column)
        for column in data.columns
        if str(column).strip().lower() in _LABEL_COLUMN_NAMES
    ]
    if label_columns:
        raise ValueError(
            "labels must not be passed to build_features; keep forward-looking "
            f"targets separate from the feature matrix: {label_columns}"
        )

    missing_columns = [
        column for column in _REQUIRED_OHLCV_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"data is missing required OHLCV columns: {missing_columns}")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build an aligned, scale-independent feature matrix without future data.

    Every value at row ``t`` uses only OHLCV observations available through
    row ``t``.  All rolling calculations are trailing, no negative shifts are
    used, and indicator warm-up periods remain ``NaN``.  Labels are rejected
    if included in ``data`` because forward-looking targets must never become
    model features.
    """
    _validate_ohlcv_input(data)

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    daily_return = close.pct_change(fill_method=None)

    ema_12 = ema(close, span=12)
    ema_26 = ema(close, span=26)
    macd_line, macd_signal, macd_histogram = macd(close)
    upper_band, _, lower_band = bollinger_bands(close)
    band_width = upper_band - lower_band
    atr_14 = atr(high, low, close, period=14)

    return pd.DataFrame(
        {
            "return_1d": daily_return,
            "return_5d": close.pct_change(5, fill_method=None),
            "return_10d": close.pct_change(10, fill_method=None),
            "ema_12_rel": (ema_12 / close) - 1,
            "ema_26_rel": (ema_26 / close) - 1,
            "rsi_14": rsi(close, period=14),
            "macd_line_rel": macd_line / close,
            "macd_signal_rel": macd_signal / close,
            "macd_histogram_rel": macd_histogram / close,
            "bollinger_percent_b": (close - lower_band) / band_width,
            "atr_14_rel": atr_14 / close,
            "realized_volatility_10d": daily_return.rolling(
                window=10, min_periods=10
            ).std(),
            "realized_volatility_20d": daily_return.rolling(
                window=20, min_periods=20
            ).std(),
            "volume_zscore_10d": (
                (volume - volume.rolling(window=10, min_periods=10).mean())
                / volume.rolling(window=10, min_periods=10).std()
            ),
            "return_1d_lag_1": daily_return.shift(1),
            "return_1d_lag_2": daily_return.shift(2),
            "return_1d_lag_3": daily_return.shift(3),
        },
        index=data.index,
    )


def build_labels(
    data: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.0
) -> pd.Series:
    """Return {-1, 0, 1} labels from a deliberately forward-looking return.

    At row ``t``, the label compares ``close[t + forward_periods]`` with
    ``close[t]``.  This future dependency is intentional for supervised
    targets, but labels must never be included in :func:`build_features` or
    used as a model input.  The final ``forward_periods`` labels are ``NaN``
    because their future prices do not exist.
    """
    _validate_ohlcv_input(data)
    if forward_periods < 1:
        raise ValueError("forward_periods must be at least 1")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    close = data["close"]
    forward_return = close.shift(-forward_periods) / close - 1
    labels = pd.Series(np.nan, index=data.index, dtype="float64", name="label")
    valid_labels = forward_return.notna()
    labels.loc[valid_labels] = 0.0
    labels.loc[valid_labels & (forward_return > threshold)] = 1.0
    labels.loc[valid_labels & (forward_return < -threshold)] = -1.0
    return labels
