"""Performance metrics for completed backtest equity curves and trades."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from src.backtest.engine import Trade


def _period_returns(equity_curve: pd.Series) -> pd.Series:
    """Return finite period returns without filling missing equity values."""
    returns = equity_curve.pct_change(fill_method=None).dropna()
    return returns[np.isfinite(returns)]


def sharpe_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the annualized Sharpe ratio using sample return volatility.

    The annual risk-free rate is converted to a per-period simple rate.  The
    function returns ``NaN`` for fewer than two valid returns or zero
    volatility, where the ratio is undefined.
    """
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    returns = _period_returns(equity_curve)
    if len(returns) < 2:
        return float("nan")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    volatility = float(excess_returns.std(ddof=1))
    if not np.isfinite(volatility) or volatility == 0:
        return float("nan")
    return float(sqrt(periods_per_year) * excess_returns.mean() / volatility)


def sortino_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the annualized Sortino ratio using all-period downside deviation.

    Downside returns are excess returns below zero, with non-negative periods
    contributing zero to the population downside deviation.  ``NaN`` is
    returned when fewer than two valid returns exist or downside deviation is
    zero, because a finite ratio is then undefined.
    """
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    returns = _period_returns(equity_curve)
    if len(returns) < 2:
        return float("nan")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = np.minimum(excess_returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))
    if not np.isfinite(downside_deviation) or downside_deviation == 0:
        return float("nan")
    return float(
        sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    )


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Return compounded annual growth rate from the first to last equity value.

    The number of compounding intervals is ``len(equity_curve) - 1``.  ``NaN``
    is returned for fewer than two observations, non-finite endpoints, or a
    non-positive starting equity.  A final equity of zero validly returns
    ``-1.0``.
    """
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    if len(equity_curve) < 2:
        return float("nan")

    starting_equity = float(equity_curve.iloc[0])
    ending_equity = float(equity_curve.iloc[-1])
    if (
        not np.isfinite(starting_equity)
        or not np.isfinite(ending_equity)
        or starting_equity <= 0
        or ending_equity < 0
    ):
        return float("nan")
    if ending_equity == 0:
        return -1.0

    periods = len(equity_curve) - 1
    return float((ending_equity / starting_equity) ** (periods_per_year / periods) - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the most negative peak-to-trough percentage drawdown.

    ``NaN`` is returned for an empty curve or when no positive running peak
    exists, because percentage drawdown is undefined in those cases.
    """
    if equity_curve.empty:
        return float("nan")
    values = equity_curve.astype(float)
    if not np.isfinite(values).all():
        return float("nan")

    running_peak = values.cummax()
    if (running_peak <= 0).all():
        return float("nan")
    drawdowns = (values / running_peak.where(running_peak > 0)) - 1
    return float(drawdowns.min())


def win_rate(trades: list[Trade]) -> float:
    """Return winners divided by completed trades; empty input returns ``NaN``.

    Trades with ``pnl=None`` or non-finite PnL are open/invalid and excluded.
    A zero-PnL completed trade counts as non-winning in the denominator.
    """
    completed_pnls = [
        trade.pnl
        for trade in trades
        if trade.pnl is not None and np.isfinite(trade.pnl)
    ]
    if not completed_pnls:
        return float("nan")
    return sum(pnl > 0 for pnl in completed_pnls) / len(completed_pnls)


def profit_factor(trades: list[Trade]) -> float:
    """Return gross profits divided by absolute gross losses.

    Empty input and a list containing only breakeven trades return ``NaN``.
    All-losing trades return ``0.0``.  A profitable list with no losses
    returns ``inf``, the conventional representation of an unbounded profit
    factor.  Open or non-finite-PnL trades are excluded.
    """
    completed_pnls = [
        trade.pnl
        for trade in trades
        if trade.pnl is not None and np.isfinite(trade.pnl)
    ]
    if not completed_pnls:
        return float("nan")

    gross_profit = sum(pnl for pnl in completed_pnls if pnl > 0)
    gross_loss = -sum(pnl for pnl in completed_pnls if pnl < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss
