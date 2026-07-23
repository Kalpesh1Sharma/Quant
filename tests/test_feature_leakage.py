from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_features, build_labels
from src.features.splitting import train_test_split_by_date


def _ohlcv(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(prices), freq="D", tz="UTC")
    close = pd.Series(prices, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.arange(len(close), dtype=float) + 1_000,
        },
        index=index,
    )


def test_features_are_invariant_to_future_data() -> None:
    prices = (100 + np.linspace(-5, 5, 50) + 2 * np.sin(np.arange(50))).tolist()
    data = _ohlcv(prices)

    full_features = build_features(data)
    prefix_features = build_features(data.iloc[:35])

    pd.testing.assert_frame_equal(full_features.iloc[:35], prefix_features)


def test_labels_use_only_forward_data_and_tail_is_nan() -> None:
    data = _ohlcv([100, 100, 110, 100, 100])

    labels = build_labels(data, forward_periods=2)

    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 0.0
    assert labels.iloc[2] == -1.0
    assert labels.iloc[-2:].isna().all()


def test_build_features_rejects_forward_labels_as_inputs() -> None:
    data = _ohlcv([100 + value for value in range(20)])
    data["label"] = build_labels(data)

    with pytest.raises(ValueError, match="labels must not be passed"):
        build_features(data)


def test_train_test_split_embargoes_boundary_rows() -> None:
    index = pd.date_range("2024-01-01", periods=14, freq="D", tz="UTC")
    features = pd.DataFrame({"feature": np.arange(14, dtype=float)}, index=index)
    labels = pd.Series(0.0, index=index, name="label")
    split_date = index[10]

    train_features, train_labels, test_features, test_labels = train_test_split_by_date(
        features, labels, split_date, embargo_periods=3
    )

    assert list(train_features.index) == list(index[:7])
    assert list(train_labels.index) == list(index[:7])
    assert set(index[7:10]).isdisjoint(train_features.index)
    assert list(test_features.index) == list(index[10:])
    assert list(test_labels.index) == list(index[10:])


def test_split_drops_nan_rows_from_both_features_and_labels() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    features = pd.DataFrame(
        {"feature_a": np.arange(12, dtype=float), "feature_b": np.arange(12, dtype=float)},
        index=index,
    )
    labels = pd.Series(0.0, index=index, name="label")
    features.loc[index[1], "feature_a"] = np.nan
    labels.loc[index[2]] = np.nan
    features.loc[index[8], "feature_b"] = np.nan
    labels.loc[index[9]] = np.nan

    train_features, train_labels, test_features, test_labels = train_test_split_by_date(
        features, labels, index[7], embargo_periods=1
    )

    for values in (train_features, train_labels, test_features, test_labels):
        assert not values.isna().any().any()
