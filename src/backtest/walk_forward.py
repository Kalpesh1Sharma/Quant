"""Leakage-safe walk-forward backtest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.strategies.base import Strategy


@dataclass(frozen=True)
class WalkForwardWindow:
    """Actual, inclusive date bounds of one disjoint train/test split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_windows(
    data: pd.DataFrame,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    step: pd.Timedelta | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling, non-overlapping train/test windows from ``data``.

    Window endpoints are the first and last actual observations in each
    slice, and are inclusive.  Test windows are always disjoint: ``step``
    defaults to ``test_period`` and smaller steps are rejected.  A final
    partial test window is included when it contains at least one bar.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("data index must be increasing and have no duplicates")
    if train_period <= pd.Timedelta(0) or test_period <= pd.Timedelta(0):
        raise ValueError("train_period and test_period must be positive")

    effective_step = test_period if step is None else step
    if effective_step <= pd.Timedelta(0):
        raise ValueError("step must be positive")
    if effective_step < test_period:
        raise ValueError("step must be at least test_period to avoid test overlap")
    if data.empty:
        return []

    index = data.index
    final_date = index[-1]
    window_start = index[0]
    windows: list[WalkForwardWindow] = []

    while window_start <= final_date:
        train_boundary = window_start + train_period
        test_boundary = train_boundary + test_period
        train_index = index[(index >= window_start) & (index < train_boundary)]
        test_index = index[(index >= train_boundary) & (index < test_boundary)]
        if train_index.empty or test_index.empty:
            break

        windows.append(
            WalkForwardWindow(
                train_start=train_index[0],
                train_end=train_index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
            )
        )
        window_start = window_start + effective_step

    return windows


def run_walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    windows: list[WalkForwardWindow],
    **backtest_kwargs: Any,
) -> list[BacktestResult]:
    """Run fresh strategies on each window's test slice, never its train slice.

    Each backtest receives only the inclusive ``test_start`` through
    ``test_end`` rows, so no training-period values can influence its signals,
    indicator warm-up, sizing, or fills.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    results: list[BacktestResult] = []
    for window in windows:
        test_data = data.loc[
            (data.index >= window.test_start) & (data.index <= window.test_end)
        ]
        if test_data.empty:
            raise ValueError("walk-forward window does not select test data")
        results.append(
            run_backtest(test_data, strategy_factory(), **backtest_kwargs)
        )
    return results
