"""Leakage-safe XGBoost signal generation with an explicit training boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features.engineering import build_features, build_labels
from src.features.splitting import train_test_split_by_date
from src.strategies.base import Strategy


_SIGNAL_VALUES = np.array([-1, 0, 1], dtype=int)


class XGBoostSignalStrategy(Strategy):
    """Wrap a pre-trained XGBoost model without performing any fitting."""

    def __init__(self, model: XGBClassifier, feature_columns: list[str]) -> None:
        """Store an already-trained model and its required feature columns.

        Training is intentionally external to this strategy so the model's
        train/test boundary remains auditable and cannot be hidden inside
        :meth:`generate_signals`.
        """
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns must not contain duplicates")

        self.model = model
        self.feature_columns = list(feature_columns)

    def _predictions_to_signals(self, predictions: np.ndarray) -> np.ndarray:
        """Decode model classes into the platform's {-1, 0, 1} signal space."""
        encoded_predictions = np.asarray(predictions, dtype=int)
        trained_signal_classes = getattr(self.model, "_quant_signal_classes", None)
        if trained_signal_classes is not None:
            classes = np.asarray(trained_signal_classes, dtype=int)
            if (
                (encoded_predictions < 0).any()
                or (encoded_predictions >= len(classes)).any()
            ):
                raise ValueError("model produced an encoded class outside its training map")
            signals = classes[encoded_predictions]
        else:
            signals = encoded_predictions

        if not np.isin(signals, _SIGNAL_VALUES).all():
            raise ValueError("model predictions must map to -1, 0, or 1 signals")
        return signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Predict aligned {-1, 0, 1} signals using only row-current features.

        :func:`build_features` is trailing-only, so every predicted row uses
        data available at or before that row.  Rows with any ``NaN`` feature
        are always flat and are never passed to XGBoost, preventing warm-up
        values from being interpreted as model inputs.
        """
        features = build_features(data)
        missing_columns = [
            column for column in self.feature_columns if column not in features.columns
        ]
        if missing_columns:
            raise ValueError(
                "build_features did not produce required model columns: "
                f"{missing_columns}"
            )

        model_features = features.loc[:, self.feature_columns]
        valid_rows = model_features.notna().all(axis=1)
        signals = pd.Series(0, index=data.index, dtype="int64", name="signal")
        if valid_rows.any():
            predictions = self.model.predict(model_features.loc[valid_rows])
            signals.loc[valid_rows] = self._predictions_to_signals(predictions)
        return signals


def _classification_metrics(
    labels: pd.Series, predictions: np.ndarray
) -> tuple[float, dict[str, dict[str, float | int]]]:
    """Calculate accuracy and per-signal-class test metrics without warnings."""
    signal_predictions = np.asarray(predictions, dtype=int)
    accuracy = float(accuracy_score(labels, signal_predictions))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        labels,
        signal_predictions,
        labels=_SIGNAL_VALUES,
        zero_division=0,
    )
    per_class = {
        str(signal): {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1_score": float(f1_score[position]),
            "support": int(support[position]),
        }
        for position, signal in enumerate(_SIGNAL_VALUES)
    }
    return accuracy, per_class


def train_xgboost_signal(
    data: pd.DataFrame,
    split_date: pd.Timestamp,
    forward_periods: int = 5,
    threshold: float = 0.0,
    embargo_periods: int = 5,
    model_params: dict[str, Any] | None = None,
) -> tuple[XGBoostSignalStrategy, dict[str, Any]]:
    """Train once on the embargoed train split and evaluate only on the test split.

    Features are trailing-only, while labels intentionally use forward prices.
    The split therefore receives ``forward_periods`` and excludes the
    pre-split embargo rows before this function calls ``model.fit``.  The
    model is fitted on ``train_features`` and ``train_labels`` only: test rows
    are never passed to ``fit`` and are used solely for held-out reporting.
    """
    features = build_features(data)
    labels = build_labels(
        data, forward_periods=forward_periods, threshold=threshold
    )
    train_features, train_labels, test_features, test_labels = train_test_split_by_date(
        features,
        labels,
        split_date,
        embargo_periods=embargo_periods,
        forward_periods=forward_periods,
    )
    if train_features.empty:
        raise ValueError("no valid training rows remain after warm-up and embargo")

    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels.astype(int))
    if len(label_encoder.classes_) < 2:
        raise ValueError("XGBoost training requires at least two signal classes")

    parameters: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "objective": "multi:softprob",
        "num_class": len(label_encoder.classes_),
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }
    if model_params is not None:
        parameters.update(model_params)
    parameters["random_state"] = 42

    model = XGBClassifier(**parameters)
    model.fit(train_features, encoded_train_labels)
    model._quant_signal_classes = label_encoder.classes_.astype(int).tolist()

    strategy = XGBoostSignalStrategy(model, list(train_features.columns))
    encoded_train_predictions = model.predict(train_features)
    train_predictions = strategy._predictions_to_signals(encoded_train_predictions)
    train_accuracy, _ = _classification_metrics(train_labels, train_predictions)

    if test_features.empty:
        test_accuracy = float("nan")
        test_per_class = {
            str(signal): {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1_score": float("nan"),
                "support": 0,
            }
            for signal in _SIGNAL_VALUES
        }
    else:
        encoded_test_predictions = model.predict(test_features)
        test_predictions = strategy._predictions_to_signals(encoded_test_predictions)
        test_accuracy, test_per_class = _classification_metrics(
            test_labels, test_predictions
        )

    metrics: dict[str, Any] = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_row_count": len(train_features),
        "test_row_count": len(test_features),
        "test_per_class": test_per_class,
        "feature_importances": {
            column: float(importance)
            for column, importance in zip(
                strategy.feature_columns, model.feature_importances_, strict=True
            )
        },
    }
    return strategy, metrics
