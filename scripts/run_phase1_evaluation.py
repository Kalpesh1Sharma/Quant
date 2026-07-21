"""Run transparent walk-forward evaluations on ten years of daily OHLCV data.

This script deliberately reports every completed evaluation, including weak or
unprofitable ones.  It uses only out-of-sample test windows: each window is
backtested independently, then its return path is linked to the prior test
window so aggregate equity does not reset to initial capital at each boundary.

KNOWN LIMITATION (documented, not fixed, for Phase 1): each walk-forward test
window is sliced independently before being handed to run_backtest, so every
window's indicators (EMA/RSI/Bollinger/ATR) warm up from scratch. This means
the first ~20-26 rows of every window are forced flat while indicators fill
their lookback, rather than carrying warmed-up indicator state across the
train/test boundary. A more correct version would extend each test-window
fetch backward by the indicator lookback and only start counting trades/equity
from the true test_start. Left as a documented simplification for Phase 1
rather than silently fixed, since it changes evaluation results and should be
a visible, deliberate choice.

DIAGNOSTIC NOTE: run_backtest leaves a position open (pnl=None) if it is still
active at the final row of a window. Because walk-forward linking marks each
window's ending equity to market (including any open, unrealized position),
equity-based metrics (Sharpe/Sortino/CAGR) can be materially influenced by
unrealized gains/losses that are NEVER counted in trade-level stats
(win_rate/profit_factor). This script estimates that unrealized contribution
per window and in aggregate so the two families of metrics can be reconciled
rather than silently disagreeing.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestResult, Trade
from src.backtest.metrics import (
    cagr,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from src.backtest.walk_forward import WalkForwardWindow, generate_windows, run_walk_forward
from src.data_layer.fetcher import DataQualityReport, fetch_ohlcv
from src.strategies.base import Strategy
from src.strategies.mean_reversion import BollingerRsiMeanReversion
from src.strategies.momentum import EmaCrossoverMomentum


TICKERS = ("SPY", "AAPL", "BTC-USD")

# Trading-day conventions differ by asset class: equities trade ~252 days/year
# (weekdays minus holidays), while crypto trades every calendar day. Using the
# wrong constant understates/overstates annualized Sharpe, Sortino, and CAGR
# for that asset -- this must be looked up per ticker, never hard-coded once
# for the whole run.
PERIODS_PER_YEAR = {
    "SPY": 252,
    "AAPL": 252,
    "BTC-USD": 365,
}

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
TRAIN_PERIOD = pd.Timedelta(days=365 * 2)
TEST_PERIOD = pd.Timedelta(days=182)

StrategyFactory = Callable[[], Strategy]


@dataclass
class AggregatedEvaluation:
    """Continuous out-of-sample outputs plus each window's raw details."""

    equity_curve: pd.Series
    trades: list[Trade]
    window_breakdown: list[dict[str, Any]]
    zero_trade_windows: int
    total_open_trade_count: int
    total_realized_pnl: float
    total_unrealized_pnl_estimate: float


def print_data_quality_summary(ticker: str, report: DataQualityReport) -> None:
    """Print quality findings before the script makes any cleaning decision."""
    counts = {
        reason: len(report.reasons.get(reason, ()))
        for reason in ("zero_volume", "nan_ohlc", "unexplained_gap")
    }
    print(
        f"[{ticker}] data-quality flags: flagged_rows={report.flagged_row_count}; "
        f"zero_volume={counts['zero_volume']}; nan_ohlc={counts['nan_ohlc']}; "
        f"unexplained_gap={counts['unexplained_gap']}"
    )
    if counts["zero_volume"]:
        print(
            f"[{ticker}] WARNING: retaining {counts['zero_volume']} zero-volume "
            "row(s); zero volume is not automatically bad data."
        )
    if counts["unexplained_gap"]:
        print(
            f"[{ticker}] NOTE: {counts['unexplained_gap']} unexplained gap date(s) "
            "are absent from the frame and are not altered."
        )


def clean_ohlcv(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Drop only flagged NaN-OHLC rows and reject any remaining OHLCV NaNs.

    Zero-volume rows are retained intentionally.  Unexplained gaps are dates
    absent from the source frame, so this function does not manufacture or
    remove any rows for them.
    """
    report = frame.attrs.get("data_quality_report")
    if not isinstance(report, DataQualityReport):
        raise ValueError(f"{ticker}: fetch result has no usable DataQualityReport")

    print_data_quality_summary(ticker, report)
    nan_ohlc_dates = pd.DatetimeIndex(report.reasons.get("nan_ohlc", ()))
    cleaned = frame.drop(index=nan_ohlc_dates, errors="ignore").copy()

    missing_columns = [
        column for column in REQUIRED_OHLCV_COLUMNS if column not in cleaned.columns
    ]
    if missing_columns:
        raise ValueError(f"{ticker}: cleaned frame is missing columns {missing_columns}")

    remaining_nan_rows = cleaned.loc[:, REQUIRED_OHLCV_COLUMNS].isna().any(axis=1)
    if remaining_nan_rows.any():
        dates = ", ".join(
            timestamp.isoformat() for timestamp in cleaned.index[remaining_nan_rows]
        )
        raise ValueError(
            f"{ticker}: NaNs remain in required OHLCV columns after cleaning: {dates}"
        )
    if cleaned.empty:
        raise ValueError(f"{ticker}: no rows remain after dropping NaN-OHLC rows")

    print(f"[{ticker}] cleaned rows: {len(frame)} -> {len(cleaned)}")
    return cleaned


def calculate_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    periods_per_year: int,
) -> dict[str, float]:
    """Calculate the project-standard metrics without rounding their values.

    ``periods_per_year`` must match the asset's actual trading frequency
    (e.g. 252 for equities, 365 for crypto) -- using the wrong constant
    silently distorts every annualized metric (Sharpe, Sortino, CAGR).
    """
    return {
        "sharpe_ratio": sharpe_ratio(equity_curve, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(equity_curve, periods_per_year=periods_per_year),
        "cagr": cagr(equity_curve, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
    }


def series_to_records(series: pd.Series) -> list[dict[str, Any]]:
    """Serialize an indexed series without rounding its stored values."""
    return [
        {"date": pd.Timestamp(date).isoformat(), "value": float(value)}
        for date, value in series.items()
    ]


def trade_to_dict(trade: Trade) -> dict[str, Any]:
    """Serialize a trade while retaining dates and exact computed values."""
    return {
        "entry_date": trade.entry_date.isoformat(),
        "exit_date": trade.exit_date.isoformat() if trade.exit_date is not None else None,
        "direction": trade.direction,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "size": trade.size,
        "pnl": trade.pnl,
    }


def aggregate_walk_forward_results(
    results: list[BacktestResult],
    windows: list[WalkForwardWindow],
    initial_capital: float,
    periods_per_year: int,
) -> AggregatedEvaluation:
    """Link independently run test windows into one continuous OOS curve.

    Each window begins with the backtest's own ``initial_capital``.  Its equity
    curve and closed-trade quantities are therefore scaled by the previous
    aggregated ending equity before concatenation.  This preserves each
    window's percentage return while avoiding a misleading capital reset at
    every walk-forward boundary.

    For each window this also estimates how much of the window's equity
    change is attributable to a position still open (unrealized) at the
    window boundary, versus realized trade P&L -- since realized P&L is what
    win_rate/profit_factor measure, but marked-to-market equity includes
    unrealized P&L too. This reconciliation is an approximation: it nets out
    entry/exit costs already embedded in each closed trade's pnl, then
    attributes whatever equity change remains unexplained to open positions.
    """
    if len(results) != len(windows):
        raise ValueError("walk-forward results and windows must have the same length")

    linked_curves: list[pd.Series] = []
    linked_trades: list[Trade] = []
    window_breakdown: list[dict[str, Any]] = []
    current_equity = initial_capital
    zero_trade_windows = 0
    total_open_trade_count = 0
    total_realized_pnl = 0.0
    total_unrealized_pnl_estimate = 0.0

    for window_number, (window, result) in enumerate(zip(windows, results, strict=True), start=1):
        raw_equity = result.equity_curve
        if raw_equity.empty:
            raise ValueError(f"walk-forward window {window_number} returned no equity")

        raw_starting_equity = float(raw_equity.iloc[0])
        if not math.isfinite(raw_starting_equity) or raw_starting_equity <= 0:
            raise ValueError(
                f"walk-forward window {window_number} has invalid starting equity "
                f"{raw_starting_equity!r}"
            )
        scale_factor = current_equity / raw_starting_equity
        linked_equity = raw_equity.astype(float) * scale_factor
        window_starting_equity = float(linked_equity.iloc[0])
        current_equity = float(linked_equity.iloc[-1])
        linked_curves.append(linked_equity)

        scaled_trades = [
            replace(
                trade,
                size=trade.size * scale_factor,
                pnl=None if trade.pnl is None else trade.pnl * scale_factor,
            )
            for trade in result.trades
        ]
        linked_trades.extend(scaled_trades)
        if not result.trades:
            zero_trade_windows += 1

        realized_pnl_total = sum(
            trade.pnl for trade in scaled_trades if trade.pnl is not None
        )
        open_trade_count = sum(trade.pnl is None for trade in scaled_trades)
        window_equity_change = current_equity - window_starting_equity
        # Equity change not explained by realized trade P&L is attributed to
        # a position still open at the window boundary (marked-to-market,
        # never counted as a win/loss in win_rate or profit_factor).
        unrealized_pnl_estimate = window_equity_change - realized_pnl_total

        total_open_trade_count += open_trade_count
        total_realized_pnl += realized_pnl_total
        total_unrealized_pnl_estimate += unrealized_pnl_estimate

        window_breakdown.append(
            {
                "window_number": window_number,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "raw_equity_start": raw_starting_equity,
                "raw_equity_end": float(raw_equity.iloc[-1]),
                "linked_equity_start": window_starting_equity,
                "linked_equity_end": current_equity,
                "scale_factor": scale_factor,
                "trade_count": len(result.trades),
                "completed_trade_count": sum(trade.pnl is not None for trade in result.trades),
                "open_trade_count": open_trade_count,
                "realized_pnl": realized_pnl_total,
                "unrealized_pnl_estimate": unrealized_pnl_estimate,
                "metrics": calculate_metrics(linked_equity, scaled_trades, periods_per_year),
                "raw_equity_curve": series_to_records(raw_equity),
                "linked_equity_curve": series_to_records(linked_equity),
                "trades": [trade_to_dict(trade) for trade in scaled_trades],
            }
        )

    continuous_equity = pd.concat(linked_curves).sort_index()
    if continuous_equity.index.has_duplicates:
        raise ValueError("walk-forward test windows produced duplicate equity dates")
    return AggregatedEvaluation(
        equity_curve=continuous_equity,
        trades=linked_trades,
        window_breakdown=window_breakdown,
        zero_trade_windows=zero_trade_windows,
        total_open_trade_count=total_open_trade_count,
        total_realized_pnl=total_realized_pnl,
        total_unrealized_pnl_estimate=total_unrealized_pnl_estimate,
    )


def run_evaluation(
    ticker: str,
    strategy_name: str,
    strategy_factory: StrategyFactory,
    scenario_name: str,
    cost_model: CostModel,
    data: pd.DataFrame,
    windows: list[WalkForwardWindow],
    periods_per_year: int,
) -> dict[str, Any]:
    """Run one strategy/cost scenario and return all aggregate and window data."""
    print(f"[{ticker}] running {strategy_name} / {scenario_name} ...")
    results = run_walk_forward(
        data,
        strategy_factory,
        windows,
        initial_capital=INITIAL_CAPITAL,
        cost_model=cost_model,
        risk_per_trade=RISK_PER_TRADE,
        atr_period=ATR_PERIOD,
    )
    aggregated = aggregate_walk_forward_results(
        results, windows, INITIAL_CAPITAL, periods_per_year
    )
    metrics = calculate_metrics(aggregated.equity_curve, aggregated.trades, periods_per_year)

    total_gain = aggregated.equity_curve.iloc[-1] - aggregated.equity_curve.iloc[0]
    unrealized_share = (
        aggregated.total_unrealized_pnl_estimate / total_gain
        if total_gain != 0
        else float("nan")
    )

    return {
        "ticker": ticker,
        "strategy": strategy_name,
        "cost_scenario": scenario_name,
        "cost_model": asdict(cost_model),
        "periods_per_year": periods_per_year,
        "window_count": len(windows),
        "zero_trade_window_count": aggregated.zero_trade_windows,
        "aggregate": {
            "equity_start": float(aggregated.equity_curve.iloc[0]),
            "equity_end": float(aggregated.equity_curve.iloc[-1]),
            "trade_count": len(aggregated.trades),
            "completed_trade_count": sum(
                trade.pnl is not None for trade in aggregated.trades
            ),
            "open_trade_count": aggregated.total_open_trade_count,
            "realized_pnl": aggregated.total_realized_pnl,
            "unrealized_pnl_estimate": aggregated.total_unrealized_pnl_estimate,
            "unrealized_pnl_share_of_total_gain": unrealized_share,
            "metrics": metrics,
            "equity_curve": series_to_records(aggregated.equity_curve),
            "trades": [trade_to_dict(trade) for trade in aggregated.trades],
        },
        "windows": aggregated.window_breakdown,
    }


def _json_value(value: Any) -> Any:
    """Convert timestamps and non-finite floats into explicit JSON values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_results(payload: dict[str, Any], evaluation_date: pd.Timestamp) -> Path:
    """Write unrounded results to a date-stamped, valid JSON file."""
    results_directory = Path("results")
    results_directory.mkdir(parents=True, exist_ok=True)
    output_path = results_directory / (
        f"phase1_evaluation_{evaluation_date.strftime('%Y-%m-%d')}.json"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_value(payload), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def print_summary(evaluations: list[dict[str, Any]]) -> None:
    """Print a compact display table while retaining unrounded saved values."""
    if not evaluations:
        print("\nNo successful strategy/cost evaluations to report.")
        return

    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        aggregate = evaluation["aggregate"]
        metrics = aggregate["metrics"]
        rows.append(
            {
                "ticker": evaluation["ticker"],
                "strategy": evaluation["strategy"],
                "cost": evaluation["cost_scenario"],
                "periods_per_year": evaluation["periods_per_year"],
                "windows": evaluation["window_count"],
                "zero_trade_windows": evaluation["zero_trade_window_count"],
                "trades": aggregate["trade_count"],
                "open_trades": aggregate["open_trade_count"],
                "sharpe": metrics["sharpe_ratio"],
                "sortino": metrics["sortino_ratio"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "unrealized_share": aggregate["unrealized_pnl_share_of_total_gain"],
            }
        )

    summary = pd.DataFrame(rows)
    metric_columns = (
        "sharpe",
        "sortino",
        "cagr",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "unrealized_share",
    )
    formatters = {
        column: (lambda value: f"{value:.4f}" if pd.notna(value) else "NaN")
        for column in metric_columns
    }
    print("\nOut-of-sample walk-forward summary (display rounded; JSON is unrounded):")
    print(summary.to_string(index=False, formatters=formatters))


def main() -> None:
    """Fetch, clean, evaluate, report, and persist Phase 1 results."""
    evaluation_end = pd.Timestamp.now(tz="UTC").normalize()
    evaluation_start = evaluation_end - pd.DateOffset(years=10)
    strategy_definitions: tuple[tuple[str, StrategyFactory], ...] = (
        ("ema_crossover_momentum", EmaCrossoverMomentum),
        ("bollinger_rsi_mean_reversion", BollingerRsiMeanReversion),
    )
    cost_scenarios: tuple[tuple[str, CostModel], ...] = (
        ("baseline_10bps", CostModel()),
        ("stress_25bps", CostModel(transaction_cost_bps=25.0)),
    )
    evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    quality_reports: dict[str, dict[str, Any]] = {}

    print(
        "Phase 1 evaluation: "
        f"{evaluation_start.date().isoformat()} through {evaluation_end.date().isoformat()}"
    )
    for ticker in TICKERS:
        periods_per_year = PERIODS_PER_YEAR.get(ticker)
        if periods_per_year is None:
            print(
                f"[{ticker}] FAILED: no periods_per_year configured for this "
                "ticker -- add it to PERIODS_PER_YEAR before evaluating."
            )
            failures.append(
                {
                    "ticker": ticker,
                    "stage": "configuration",
                    "error": "missing PERIODS_PER_YEAR entry",
                }
            )
            continue

        print(f"\n[{ticker}] fetching daily OHLCV data ...")
        try:
            fetched = fetch_ohlcv(
                ticker,
                evaluation_start.date().isoformat(),
                evaluation_end.date().isoformat(),
                interval="1d",
            )
            report = fetched.attrs["data_quality_report"]
            if not isinstance(report, DataQualityReport):
                raise ValueError("fetcher returned an unexpected data-quality report")
            quality_reports[ticker] = {
                "flagged_row_count": report.flagged_row_count,
                "flagged_dates": [date.isoformat() for date in report.flagged_dates],
                "reasons": {
                    reason: [date.isoformat() for date in dates]
                    for reason, dates in report.reasons.items()
                },
            }
            cleaned = clean_ohlcv(ticker, fetched)
            windows = generate_windows(
                cleaned,
                train_period=TRAIN_PERIOD,
                test_period=TEST_PERIOD,
                step=TEST_PERIOD,
            )
            if not windows:
                raise ValueError("insufficient clean history for a 2-year/6-month window")
            print(f"[{ticker}] generated {len(windows)} non-overlapping test window(s).")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"[{ticker}] FAILED during fetch/clean/window generation: {message}")
            failures.append({"ticker": ticker, "stage": "fetch_clean_windows", "error": message})
            continue

        for strategy_name, strategy_factory in strategy_definitions:
            for scenario_name, cost_model in cost_scenarios:
                try:
                    evaluations.append(
                        run_evaluation(
                            ticker,
                            strategy_name,
                            strategy_factory,
                            scenario_name,
                            cost_model,
                            cleaned,
                            windows,
                            periods_per_year,
                        )
                    )
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    print(
                        f"[{ticker}] FAILED {strategy_name} / {scenario_name}: {message}"
                    )
                    failures.append(
                        {
                            "ticker": ticker,
                            "strategy": strategy_name,
                            "cost_scenario": scenario_name,
                            "stage": "backtest",
                            "error": message,
                        }
                    )

    print_summary(evaluations)
    payload = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "requested_start": evaluation_start.isoformat(),
        "requested_end": evaluation_end.isoformat(),
        "configuration": {
            "tickers": list(TICKERS),
            "periods_per_year": PERIODS_PER_YEAR,
            "interval": "1d",
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade": RISK_PER_TRADE,
            "atr_period": ATR_PERIOD,
            "train_period": str(TRAIN_PERIOD),
            "test_period": str(TEST_PERIOD),
            "step": str(TEST_PERIOD),
        },
        "known_limitations": [
            "Each walk-forward test window recomputes indicators from a cold "
            "start (no carried-over history from the training window or prior "
            "test window), so the first ~20-26 rows of every window are "
            "flat/untradeable while indicators warm up. See module docstring.",
            "unrealized_pnl_estimate/unrealized_pnl_share_of_total_gain are "
            "approximations reconciling marked-to-market equity against "
            "realized trade P&L; they are not exact accounting, since they "
            "net out whatever equity change is unexplained by closed trades "
            "and attribute it to open positions at window boundaries.",
        ],
        "data_quality_reports": quality_reports,
        "evaluations": evaluations,
        "failures": failures,
    }
    output_path = save_results(payload, evaluation_end)
    print(f"\nSaved complete, unrounded evaluation data to {output_path}")

    if failures:
        print("\nFailures (other evaluations continued):")
        for failure in failures:
            context = " / ".join(
                failure[key]
                for key in ("ticker", "strategy", "cost_scenario")
                if key in failure
            )
            print(f"- {context} [{failure['stage']}]: {failure['error']}")
    else:
        print("\nCompleted without fetch or evaluation failures.")


if __name__ == "__main__":
    main()