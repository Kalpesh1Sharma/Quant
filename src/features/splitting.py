"""Date-based ML train/test splitting with forward-label embargo protection."""

from __future__ import annotations

import pandas as pd


def _coerce_split_date(split_date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Put ``split_date`` in the index timezone before date comparisons."""
    timestamp = pd.Timestamp(split_date)
    if index.tz is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(index.tz)
    return timestamp.tz_convert(index.tz)


def train_test_split_by_date(
    features: pd.DataFrame,
    labels: pd.Series,
    split_date: pd.Timestamp,
    embargo_periods: int = 5,
    forward_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split features and labels by date while embargoing pre-split labels.

    Labels encode forward returns, so rows immediately before ``split_date``
    can have targets derived from prices on the test side. The final
    ``embargo_periods`` pre-split rows are excluded from training to prevent
    that target leakage. Rows with any feature ``NaN`` or a ``NaN`` label are
    then explicitly dropped from both sets; no imputation or implicit model
    failure is allowed.

    ``forward_periods``, if given, must be the same value used to build
    ``labels`` via ``build_labels``. This is validated explicitly: if
    ``embargo_periods < forward_periods``, a training-row label near the
    split boundary could still be computed from test-side price data, since
    ``embargo_periods`` and the labels' own forward-looking window are two
    independent numbers with no relationship enforced elsewhere. Passing
    ``forward_periods`` lets this function catch that mismatch instead of
    silently leaking. If ``forward_periods`` is omitted, no such check is
    performed -- callers who skip it are responsible for ensuring
    ``embargo_periods`` is large enough themselves.
    """
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must have identical indexes")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features and labels must use a DatetimeIndex")
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("features index must be increasing and have no duplicates")
    if embargo_periods < 0:
        raise ValueError("embargo_periods must be non-negative")
    if features.empty:
        raise ValueError("features and labels must not be empty")
    if forward_periods is not None and embargo_periods < forward_periods:
        raise ValueError(
            f"embargo_periods ({embargo_periods}) must be >= forward_periods "
            f"({forward_periods}) used to build labels, or training labels "
            "near the split boundary may depend on test-side price data"
        )

    boundary = _coerce_split_date(split_date, features.index)
    if boundary < features.index.min() or boundary > features.index.max():
        raise ValueError("split_date must fall within the features index range")

    pre_split_features = features.loc[features.index < boundary]
    pre_split_labels = labels.loc[labels.index < boundary]
    if embargo_periods:
        train_features = pre_split_features.iloc[:-embargo_periods]
        train_labels = pre_split_labels.iloc[:-embargo_periods]
    else:
        train_features = pre_split_features
        train_labels = pre_split_labels

    if train_features.empty:
        raise ValueError("embargo removes all training data")

    test_features = features.loc[features.index >= boundary]
    test_labels = labels.loc[labels.index >= boundary]

    train_valid = train_features.notna().all(axis=1) & train_labels.notna()
    test_valid = test_features.notna().all(axis=1) & test_labels.notna()
    return (
        train_features.loc[train_valid],
        train_labels.loc[train_valid],
        test_features.loc[test_valid],
        test_labels.loc[test_valid],
    )