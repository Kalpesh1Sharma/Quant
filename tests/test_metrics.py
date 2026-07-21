from __future__ import annotations

from math import isnan

import pandas as pd
import pytest

from src.backtest.engine import Trade
from src.backtest.metrics import (
    cagr,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _trade(pnl: float | None) -> Trade:
    date = pd.Timestamp("2024-01-01", tz="UTC")
    return Trade(
        entry_date=date,
        exit_date=date if pnl is not None else None,
        direction=1,
        entry_price=100.0,
        exit_price=100.0 if pnl is not None else None,
        size=1.0,
        pnl=pnl,
    )


def test_equity_metrics_match_hand_calculated_values() -> None:
    # Returns are +10%, -10%, +10%. Their sample Sharpe is sqrt(3) / 6;
    # their all-period downside-deviation Sortino is sqrt(3) / 3.
    equity_curve = pd.Series([100.0, 110.0, 99.0, 108.9])

    assert sharpe_ratio(equity_curve, periods_per_year=1) == pytest.approx(0.2886751346)
    assert sortino_ratio(equity_curve, periods_per_year=1) == pytest.approx(0.5773502692)
    assert cagr(equity_curve, periods_per_year=1) == pytest.approx(0.0288276478)
    assert max_drawdown(equity_curve) == pytest.approx(-0.10)


def test_trade_metrics_match_hand_calculated_winners_and_losers() -> None:
    trades = [_trade(100.0), _trade(-40.0), _trade(0.0)]

    assert win_rate(trades) == pytest.approx(1 / 3)
    assert profit_factor(trades) == pytest.approx(2.5)


def test_metrics_handle_empty_flat_and_all_losing_inputs() -> None:
    flat_curve = pd.Series([100.0, 100.0, 100.0])

    assert isnan(sharpe_ratio(flat_curve))
    assert isnan(sortino_ratio(flat_curve))
    assert cagr(flat_curve) == 0.0
    assert max_drawdown(flat_curve) == 0.0
    assert isnan(win_rate([]))
    assert isnan(profit_factor([]))
    assert win_rate([_trade(-10.0), _trade(-5.0)]) == 0.0
    assert profit_factor([_trade(-10.0), _trade(-5.0)]) == 0.0
