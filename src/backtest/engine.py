"""Leakage-safe, bar-by-bar backtesting with next-open execution.

Signals generated from data through row ``t`` are executed only at row
``t + 1``'s open.  In particular, the engine never fills an order at the same
bar's close that produced the signal.  ATR used for sizing at that fill is
also taken from row ``t``, the latest completed bar available at the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from src.backtest.costs import CostModel, apply_costs
from src.indicators.atr import atr
from src.position_sizing import fixed_fractional_size
from src.strategies.base import Strategy


@dataclass
class Trade:
    """A filled position, with exit fields unset while the trade is open."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    direction: int
    entry_price: float
    exit_price: float | None
    size: float
    pnl: float | None


@dataclass
class BacktestResult:
    """Outputs from one independent backtest simulation."""

    equity_curve: pd.Series
    trades: list[Trade]
    position_history: pd.Series


_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_data(data: pd.DataFrame) -> None:
    """Validate the price data assumptions needed for deterministic fills."""
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"data is missing required columns: {missing_columns}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("data index must be increasing and have no duplicates")
    if data.loc[:, _REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("data contains NaN OHLCV values; fills would be undefined")


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    cost_model: CostModel = CostModel(),
    risk_per_trade: float = 0.01,
    atr_period: int = 14,
) -> BacktestResult:
    """Simulate ``strategy`` one bar at a time using next-bar-open fills.

    A signal at row ``t`` is submitted at row ``t + 1``'s open, never the
    signal bar's close.  Position sizing uses ATR from row ``t`` and equity
    available immediately before the new position is entered.  Existing
    positions are marked to each bar's close.  An open position at the final
    row is intentionally left open (``exit_date`` and ``pnl`` are ``None``)
    rather than being given an unrealistic final close fill.

    Input OHLCV values must be complete and finite.  Invalid data is rejected
    explicitly because a missing price cannot produce an auditable fill.
    """
    _validate_data(data)
    if not isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if not isfinite(risk_per_trade) or risk_per_trade < 0:
        raise ValueError("risk_per_trade must be finite and non-negative")
    if atr_period < 1:
        raise ValueError("atr_period must be at least 1")

    signals = strategy.generate_signals(data)
    if not signals.index.equals(data.index):
        raise ValueError("strategy signals must have exactly data.index")
    if not signals.isin([-1, 0, 1]).all():
        raise ValueError("strategy signals must contain only -1, 0, or 1")

    atr_values = atr(data["high"], data["low"], data["close"], period=atr_period)
    equity_values: list[float] = []
    position_values: list[float] = []
    trades: list[Trade] = []

    cash = float(initial_capital)
    position_size = 0.0
    position_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    active_trade: Trade | None = None

    for row_position, (date, row) in enumerate(data.iterrows()):
        if row_position > 0:
            desired_direction = int(signals.iloc[row_position - 1])
            execution_atr = float(atr_values.iloc[row_position - 1])
            open_price = float(row["open"])

            if desired_direction != position_direction:
                if position_direction != 0:
                    exit_fill_direction = -position_direction
                    effective_exit_price = apply_costs(
                        open_price, entry_atr, cost_model, exit_fill_direction
                    )
                    if position_direction == 1:
                        cash += position_size * effective_exit_price
                    else:
                        cash -= position_size * effective_exit_price

                    pnl = position_direction * (
                        effective_exit_price - entry_price
                    ) * position_size
                    if active_trade is None:
                        raise RuntimeError("open position has no active trade")
                    active_trade.exit_date = date
                    active_trade.exit_price = effective_exit_price
                    active_trade.pnl = pnl
                    active_trade = None
                    position_size = 0.0
                    position_direction = 0

                if desired_direction != 0:
                    new_size = fixed_fractional_size(
                        equity=cash,
                        risk_per_trade=risk_per_trade,
                        entry_price=open_price,
                        atr_at_entry=execution_atr,
                    )
                    if new_size > 0:
                        effective_entry_price = apply_costs(
                            open_price,
                            execution_atr,
                            cost_model,
                            desired_direction,
                        )
                        if desired_direction == 1:
                            cash -= new_size * effective_entry_price
                        else:
                            cash += new_size * effective_entry_price

                        position_size = new_size
                        position_direction = desired_direction
                        entry_price = effective_entry_price
                        entry_atr = execution_atr
                        active_trade = Trade(
                            entry_date=date,
                            exit_date=None,
                            direction=desired_direction,
                            entry_price=effective_entry_price,
                            exit_price=None,
                            size=new_size,
                            pnl=None,
                        )
                        trades.append(active_trade)

        close_price = float(row["close"])
        marked_equity = cash + (position_direction * position_size * close_price)
        equity_values.append(marked_equity)
        position_values.append(position_direction * position_size)

    return BacktestResult(
        equity_curve=pd.Series(equity_values, index=data.index, name="equity"),
        trades=trades,
        position_history=pd.Series(
            position_values, index=data.index, name="position", dtype="float64"
        ),
    )
