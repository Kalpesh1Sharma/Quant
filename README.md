# AI Quant Trading Platform — Phase 1

A backtesting research platform built around one central principle: **a
backtest is only as trustworthy as its leakage-safety and its honesty about
what its own numbers mean.** Phase 1 covers the data layer, technical
indicators, two trading strategies, a bar-by-bar backtesting engine with
walk-forward validation, and an evaluation run against 10 years of real
market data.

This README reports what was actually found — including a strategy that
looked strong on paper but wasn't, and a methodological trap that would have
gone unnoticed without an explicit diagnostic built to catch it.

## What's implemented

- **Data layer** (`src/data_layer/`): fetches and locally caches daily OHLCV
  data (via `yfinance`), with an NYSE trading-calendar-aware gap detector and
  a non-destructive data-quality report (flags zero-volume rows, NaN OHLC
  values, and unexplained calendar gaps without silently dropping them).
- **Indicators** (`src/indicators/`): EMA, RSI, MACD, Bollinger Bands, ATR —
  hand-implemented rather than pulled from a library, with RSI and ATR using
  true Wilder's smoothing (simple-average seed, then recursive smoothing).
- **Strategies** (`src/strategies/`): an EMA-crossover momentum strategy and
  a Bollinger/RSI mean-reversion strategy, both built against a `Strategy`
  interface whose leakage-safety is verified by tests that assert a signal
  never changes when the strategy is truncated to an earlier subset of data.
- **Backtest engine** (`src/backtest/`): bar-by-bar simulation with
  next-bar-open execution (a signal generated from data through bar *t* is
  never filled at bar *t*'s own close — only at bar *t+1*'s open), ATR-based
  fixed-fractional position sizing, and a parameterized transaction-cost
  model.
- **Walk-forward validation** (`src/backtest/walk_forward.py`): rolling,
  strictly non-overlapping train/test windows; only test-window performance
  is ever reported.
- **Metrics** (`src/backtest/metrics.py`): Sharpe, Sortino, CAGR, max
  drawdown, win rate, profit factor — each with explicit, documented
  behavior for edge cases (empty trade lists, zero-volatility curves,
  all-losing trades) rather than silent NaN propagation or crashes.

## Methodology

- **Data:** 10 years of daily bars for SPY, AAPL, and BTC-USD, fetched
  directly from Yahoo Finance.
- **Cleaning:** rows flagged as having NaN OHLC values are dropped before
  backtesting (an undefined price can't produce a defined fill). Zero-volume
  rows are intentionally *kept* — zero volume isn't automatically bad data.
  Calendar gaps are left untouched, since they're simply dates absent from
  the source, not bad rows to fix.
- **Walk-forward windows:** 2-year training period, 6-month non-overlapping
  test period, over the full 10-year history — 17 windows per asset. Only
  test-window results are ever used for reported metrics; training windows
  never contribute to a reported number.
- **Costs:** every combination was run twice — once at a 10 bps baseline
  transaction cost, once at a 25 bps stress scenario — specifically to
  surface how much of any apparent edge survives realistic friction.
- **Position sizing:** fixed-fractional, sized so that an ATR-based stop
  distance risks 1% of current equity per trade.
- **Linking across windows:** each window's equity curve is independently
  simulated starting from a fixed initial capital, then scaled and chained
  onto the previous window's ending equity — this preserves each window's
  true percentage return while avoiding a misleading capital reset at every
  6-month boundary.

## The main finding: a strategy's Sharpe ratio can be almost entirely fake

The backtest engine leaves a position open (unrealized) if it's still active
at the last bar of a test window, rather than forcing an artificial close.
This is the right behavior for realism — but it creates a subtle trap:
**equity-curve-based metrics (Sharpe, Sortino, CAGR) mark unrealized
positions to market, while trade-level metrics (win rate, profit factor)
only count trades that have actually closed.** A strategy can show a strong
equity curve almost entirely because of favorable unrealized positions at
window boundaries, while every trade it has actually *closed* lost money.

To catch this, an unrealized-P&L reconciliation diagnostic was added: for
each window, the equity change unexplained by realized (closed) trade P&L is
attributed to open positions, and rolled up across the full evaluation.

**This caught something real.** Across all three assets, the EMA-crossover
momentum strategy's closed-trade statistics tell a consistently different
story than its equity curve:

| Ticker  | Sharpe (baseline) | Win rate | Profit factor |
|---------|-------------------|----------|----------------|
| SPY     | 0.10               | 25.0%    | 0.37           |
| AAPL    | 0.40               | 15.9%    | 0.30           |
| BTC-USD | 1.10               | 25.4%    | 0.71           |

Every single momentum result has a win rate under 26% and a profit factor
under 1 — meaning realized losses exceed realized gains on every closed
trade, on every asset, under every cost scenario tested. The positive
Sharpe ratios above are driven substantially or (for BTC-USD, where the
unrealized share of total gain exceeds 100%) *entirely* by unrealized
open-position marks at window boundaries, not by a demonstrated trading
edge. Without the reconciliation diagnostic, this would have read as three
solid, profitable results.

## Honest results by asset and strategy

*(baseline = 10 bps transaction cost, stress = 25 bps)*

| Ticker  | Strategy       | Cost     | Sharpe | Win rate | Profit factor | Verdict |
|---------|----------------|----------|--------|----------|----------------|---------|
| SPY     | Momentum       | baseline | 0.10   | 25.0%    | 0.37           | No real edge — see above |
| SPY     | Momentum       | stress   | -0.02  | 22.7%    | 0.30           | Negative under cost |
| SPY     | Mean reversion | baseline | -0.03  | 31.6%    | 0.91           | Unprofitable |
| SPY     | Mean reversion | stress   | -0.42  | 26.3%    | 0.43           | Unprofitable |
| AAPL    | Momentum       | baseline | 0.40   | 15.9%    | 0.30           | No real edge — see above |
| AAPL    | Momentum       | stress   | 0.33   | 15.9%    | 0.27           | No real edge — see above |
| AAPL    | Mean reversion | baseline | 0.29   | 57.8%    | 1.53           | **Best result — real, if thin, edge** |
| AAPL    | Mean reversion | stress   | 0.01   | 53.3%    | 1.06           | Edge nearly erased by cost |
| BTC-USD | Momentum       | baseline | 1.10   | 25.4%    | 0.71           | No real edge — see above |
| BTC-USD | Momentum       | stress   | 1.06   | 23.9%    | 0.67           | No real edge — see above |
| BTC-USD | Mean reversion | baseline | -0.75  | 51.6%    | 0.46           | Unprofitable |
| BTC-USD | Mean reversion | stress   | -0.86  | 43.6%    | 0.40           | Unprofitable |

**The one genuinely defensible result:** AAPL mean reversion (Bollinger
Bands + RSI) is profitable at realistic cost (profit factor 1.53, win rate
57.8%) and, unlike the momentum results, its edge is backed by real closed
trades rather than unrealized marks. It's a thin edge — it nearly vanishes
under the 25 bps stress scenario — but it's real. Every other
strategy/asset combination tested here is either unprofitable outright or
depends on an accounting artifact rather than a demonstrated trading edge.

This is reported as the actual finding of Phase 1, not softened: **the
platform did not discover a robust, general trading edge.** What it did
demonstrate is a rigorous, leakage-safe evaluation pipeline capable of
telling the difference between a real edge and a mirage — which is the
harder and more valuable engineering problem.

## Known limitations

- **Indicator cold-start per window:** each walk-forward test window
  computes its indicators independently, with no carried-over history from
  the training window. This forces roughly the first 20-26 rows of every
  window to be flat while indicators warm up — real evaluation time lost
  across 17 windows per asset. A more complete version would extend each
  window's data backward by the indicator lookback and only start counting
  trades/equity from the true test-window start.
- **Exit-side slippage uses entry-time ATR**, not ATR at the moment of exit.
  If volatility regime changes meaningfully between entry and exit, this
  under- or overstates exit slippage. A deliberate simplification, not an
  oversight — worth revisiting if slippage sensitivity becomes a focus.
- **The unrealized-P&L reconciliation is an approximation**, not exact
  accounting — it nets out costs already embedded in closed-trade P&L and
  attributes whatever equity change is left over to open positions. It
  becomes numerically unstable (large or meaningless ratios) when a
  window's total gain is near zero, so it's most informative for large,
  unambiguous gains and should be read alongside win rate/profit factor
  rather than in isolation.
- **Only two strategies and three assets were tested.** These results
  characterize this specific strategy design and cost model, not momentum
  or mean-reversion trading in general.

## What's next (Phase 2+)

- ML-based signal generation (XGBoost/LightGBM) as an alternative to
  hand-coded indicator rules
- Market-regime detection and strategy selection
- A proper risk-management layer (dynamic stops, portfolio-level
  diversification, max-drawdown circuit breakers)
- Paper trading loop
- Dashboard with live signals and the AI-explanation layer

## Reproducing these results

```bash
python scripts/run_phase1_evaluation.py
```

Full, unrounded results — including the per-window breakdown behind every
aggregate number in the tables above — are saved to
`results/phase1_evaluation_{date}.json`.
