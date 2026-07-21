from __future__ import annotations

from math import nan

import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import run_backtest
from src.position_sizing import fixed_fractional_size
from src.strategies.base import Strategy


def _market_data(opens: list[float], closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": [open_price + 5.0 for open_price in opens],
            "low": [open_price - 5.0 for open_price in opens],
            "close": closes,
            "volume": 1_000,
        },
        index=index,
    )


class _SingleTradeStrategy(Strategy):
    """Enter from the first signal and exit from the second signal."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index, dtype="int64")
        signals.iloc[0] = 1
        return signals


@pytest.mark.parametrize("invalid_atr", [nan, 0.0, -1.0])
def test_position_sizing_returns_zero_for_invalid_atr(invalid_atr: float) -> None:
    assert fixed_fractional_size(100_000, 0.01, 100.0, invalid_atr) == 0.0


def test_transaction_costs_reduce_final_equity_for_identical_trades() -> None:
    data = _market_data([100.0, 100.0, 120.0], [100.0, 100.0, 120.0])
    strategy = _SingleTradeStrategy()

    cost_free = run_backtest(
        data, strategy, cost_model=CostModel(transaction_cost_bps=0.0), atr_period=1
    )
    with_costs = run_backtest(
        data, strategy, cost_model=CostModel(transaction_cost_bps=10.0), atr_period=1
    )

    assert with_costs.equity_curve.iloc[-1] < cost_free.equity_curve.iloc[-1]


def test_single_long_trade_has_hand_computed_fill_size_and_pnl() -> None:
    data = _market_data([100.0, 100.0, 120.0], [100.0, 100.0, 120.0])

    result = run_backtest(
        data,
        _SingleTradeStrategy(),
        initial_capital=100_000.0,
        cost_model=CostModel(transaction_cost_bps=0.0),
        risk_per_trade=0.01,
        atr_period=1,
    )

    trade = result.trades[0]
    assert trade.entry_price == 100.0
    assert trade.exit_price == 120.0
    assert trade.size == 50.0
    assert trade.pnl == 1_000.0
    assert result.equity_curve.iloc[-1] == 101_000.0
