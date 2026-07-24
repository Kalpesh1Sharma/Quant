from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import build_features
from src.strategies.xgboost_signal import train_xgboost_signal


def _ohlcv(length: int = 180) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=length, freq="D", tz="UTC")
    positions = np.arange(length)
    close = pd.Series(
        100.0 + (0.05 * positions) + (6.0 * np.sin(positions * 0.35)),
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1_000.0 + (20.0 * np.cos(positions * 0.2)) + positions,
        },
        index=index,
    )


def _train(data: pd.DataFrame):
    return train_xgboost_signal(
        data,
        split_date=data.index[120],
        forward_periods=5,
        threshold=0.002,
        embargo_periods=5,
        model_params={"n_estimators": 20, "max_depth": 2},
    )


def test_signals_are_only_negative_one_zero_or_one() -> None:
    data = _ohlcv()
    strategy, _ = _train(data)

    signals = strategy.generate_signals(data)

    assert signals.isin([-1, 0, 1]).all()


def test_signal_is_flat_for_rows_with_nan_features() -> None:
    data = _ohlcv()
    strategy, _ = _train(data)
    features = build_features(data)

    signals = strategy.generate_signals(data)

    invalid_rows = features.isna().any(axis=1)
    assert invalid_rows.any()
    assert (signals.loc[invalid_rows] == 0).all()


def test_signal_index_matches_input_index() -> None:
    data = _ohlcv()
    strategy, _ = _train(data)

    signals = strategy.generate_signals(data.iloc[20:])

    assert signals.index.equals(data.index[20:])


def test_training_is_invariant_to_test_only_price_shock() -> None:
    data = _ohlcv()
    baseline_strategy, _ = _train(data)
    split_date = data.index[120]

    shocked = data.copy()
    test_rows = shocked.index >= split_date
    shocked.loc[test_rows, ["open", "high", "low", "close"]] *= 10.0
    shocked_strategy, _ = _train(shocked)

    training_prefix = data.loc[data.index < split_date]
    baseline_signals = baseline_strategy.generate_signals(training_prefix)
    shocked_signals = shocked_strategy.generate_signals(training_prefix)

    pd.testing.assert_series_equal(baseline_signals, shocked_signals)


def test_returned_metrics_have_expected_keys_and_ranges() -> None:
    data = _ohlcv()
    strategy, metrics = _train(data)

    expected_keys = {
        "train_accuracy",
        "test_accuracy",
        "train_row_count",
        "test_row_count",
        "test_per_class",
        "feature_importances",
    }
    assert expected_keys.issubset(metrics)
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["train_row_count"] > 0
    assert metrics["test_row_count"] > 0
    assert set(metrics["feature_importances"]) == set(build_features(data).columns)
    assert set(metrics["test_per_class"]) == {"-1", "0", "1"}
    assert strategy.feature_columns == list(build_features(data).columns)
