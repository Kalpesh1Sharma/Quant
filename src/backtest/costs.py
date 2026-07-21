"""Transaction-cost and slippage modelling for backtest fills."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CostModel:
    """Costs charged for every fill.

    ``transaction_cost_bps`` is applied to the fill price, while
    ``slippage_atr_fraction`` is an absolute price adjustment equal to that
    fraction of the ATR supplied for the fill.
    """

    transaction_cost_bps: float = 10.0
    slippage_atr_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Reject cost settings that would improve fills or are non-finite."""
        if not isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        if not isfinite(self.slippage_atr_fraction) or self.slippage_atr_fraction < 0:
            raise ValueError("slippage_atr_fraction must be finite and non-negative")


def apply_costs(
    fill_price: float,
    atr_at_entry: float | None,
    cost_model: CostModel,
    direction: int,
) -> float:
    """Return a fill price worsened by transaction costs and slippage.

    ``direction`` is ``1`` for a buy and ``-1`` for a sell.  Both cost terms
    move buys upward and sells downward, so they can never improve a trade.
    Missing, non-finite, or non-positive ATR produces zero ATR-based
    slippage; percentage transaction costs still apply.
    """
    if not isfinite(fill_price) or fill_price <= 0:
        raise ValueError("fill_price must be finite and positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be 1 (buy) or -1 (sell)")

    transaction_cost = fill_price * (cost_model.transaction_cost_bps / 10_000)
    slippage = 0.0
    if atr_at_entry is not None and isfinite(atr_at_entry) and atr_at_entry > 0:
        slippage = cost_model.slippage_atr_fraction * atr_at_entry

    return fill_price + direction * (transaction_cost + slippage)
