"""Position-sizing functions."""

from __future__ import annotations

from math import isfinite


def fixed_fractional_size(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    atr_at_entry: float,
    atr_stop_multiple: float = 2.0,
) -> float:
    """Return the units whose ATR stop risks ``risk_per_trade`` of equity.

    The stop distance is ``atr_stop_multiple * atr_at_entry``.  This function
    returns ``0.0`` rather than raising when ATR is NaN, zero, or negative.
    It also returns ``0.0`` for non-finite or non-positive equity, risk,
    entry price, or stop multiple because no defined positive position can be
    calculated from those inputs.
    """
    inputs = (equity, risk_per_trade, entry_price, atr_at_entry, atr_stop_multiple)
    if not all(isfinite(value) for value in inputs):
        return 0.0
    if (
        equity <= 0
        or risk_per_trade <= 0
        or entry_price <= 0
        or atr_at_entry <= 0
        or atr_stop_multiple <= 0
    ):
        return 0.0

    risk_amount = equity * risk_per_trade
    stop_distance = atr_stop_multiple * atr_at_entry
    return risk_amount / stop_distance
