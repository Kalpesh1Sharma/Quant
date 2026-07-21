from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import run_backtest
from src.backtest.walk_forward import generate_windows
from src.strategies.base import Strategy
from src.strategies.momentum import EmaCrossoverMomentum


def _market_data(length: int = 30) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D", tz="UTC", name="date")
    close = pd.Series([100.0 + position for position in range(length)], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": 1_000,
        },
        index=index,
    )


class _ScheduledStrategy(Strategy):
    """A deliberately trailing-only strategy used to isolate engine timing."""

    def __init__(self, long_from: int, flat_from: int) -> None:
        self.long_from = long_from
        self.flat_from = flat_from

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index, dtype="int64")
        signals.iloc[self.long_from : self.flat_from] = 1
        return signals


def test_signal_at_t_does_not_use_row_t_plus_1_or_later() -> None:
    data = _market_data()
    strategy = _ScheduledStrategy(long_from=2, flat_from=7)
    baseline = run_backtest(
        data, strategy, cost_model=CostModel(0.0), atr_period=2
    )

    changed_future = data.copy()
    changed_future.loc[changed_future.index[15] :, ["open", "high", "low", "close"]] *= 10
    changed = run_backtest(
        changed_future, strategy, cost_model=CostModel(0.0), atr_period=2
    )

    pd.testing.assert_series_equal(
        baseline.equity_curve.iloc[:15], changed.equity_curve.iloc[:15]
    )
    assert [asdict(trade) for trade in baseline.trades] == [
        asdict(trade) for trade in changed.trades
    ]


def test_execution_happens_at_next_bar_open_not_current_close() -> None:
    data = _market_data(length=4)
    data.loc[data.index[0], "close"] = 10.0
    data.loc[data.index[1], "open"] = 123.0
    strategy = _ScheduledStrategy(long_from=0, flat_from=3)

    result = run_backtest(data, strategy, cost_model=CostModel(0.0), atr_period=1)

    assert result.trades[0].entry_date == data.index[1]
    assert result.trades[0].entry_price == 123.0
    assert result.trades[0].entry_price != data.loc[data.index[0], "close"]


def test_walk_forward_train_and_test_windows_never_overlap() -> None:
    data = _market_data(length=45)
    windows = generate_windows(
        data,
        train_period=pd.Timedelta(days=10),
        test_period=pd.Timedelta(days=5),
    )

    assert windows
    for window in windows:
        assert window.train_end < window.test_start

    for earlier, later in zip(windows, windows[1:]):
        assert earlier.test_end < later.test_start


def test_no_trades_generated_during_indicator_warmup() -> None:
    data = _market_data(length=20)
    strategy = EmaCrossoverMomentum(fast_span=2, slow_span=4)

    result = run_backtest(data, strategy, cost_model=CostModel(0.0), atr_period=1)

    warmup_dates = set(data.index[:3])
    assert all(trade.entry_date not in warmup_dates for trade in result.trades)
