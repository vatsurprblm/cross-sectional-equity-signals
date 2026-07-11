# Project: Cross-Sectional ML Equity Signal Research

## Goal
Build a monthly stock-ranking model for US equities using fundamental + technical
features, validated with proper time-series methodology, and rigorously test
whether the model's signal is genuinely new information or just a repackaged
version of known factors (momentum, value, size, quality).

This is a portfolio project for UK master's applications (UCL CSML, UCL
Computational Finance, Warwick MSc Statistics - Finance Route, LSE, Imperial)
targeting a career as a trading pod quant researcher / ML quant. The project
needs to demonstrate research rigor, not just a working ML pipeline.

## Important: explain things simply
I (Yash) am new to quant finance and ML concepts. When proposing code or
explaining results, use plain language and briefly explain any jargon the
first time it comes up (e.g. "regularization," "cointegration," "walk-forward
validation"). Don't assume prior familiarity.

## Data
- `scratch/income_annual_all.csv` and `scratch/income_quarterly_all.csv` —
  SimFin free-tier fundamentals data. Covers ~4,390 US companies, but only
  2020-08 to 2025-06 (free tier limitation — no earlier history available).
  Kept in `scratch/` and gitignored, not `data/`, out of caution around
  SimFin's redistribution terms.
- Key columns: Ticker, Report Date (period the numbers describe), Publish Date
  (when the numbers actually became public — USE THIS for point-in-time
  correctness, never Report Date, to avoid look-ahead bias).
  - Known data quality issue to handle when this gets used in Step 3: 35 rows
    have a Publish Date earlier than the Report Date, which shouldn't be
    possible (includes ticker AY, which has an obviously-wrong placeholder
    Publish Date). Filter or otherwise handle before doing point-in-time joins.
- Price data: not yet pulled — will need daily OHLCV via yfinance for the
  same universe.
- Universe: historical S&P 500 constituents from
  https://github.com/fja05680/sp500 (MIT-licensed; derived file kept at
  `data/sp500/sp500_ticker_start_end.csv`, see `data/sp500/SOURCE.md`),
  cross-referenced against which tickers actually have SimFin data. Built by
  `scripts/build_universe.py` into `data/monthly_universe.csv`. Turned out to
  be ~380-390 companies/month (449 unique tickers over 2020-2025), well above
  the original ~100-150 estimate. One manual ticker crosswalk is applied
  (BRK.B → SimFin's BRK-A, since SimFin only covers Berkshire's A shares);
  10 other dot-notation tickers in the S&P 500 source have no SimFin coverage
  under any ticker and are legitimately excluded.

## Project timeline / scope (adjusted from original plan)
- Original plan was 2010-2024 for regime diversity; SimFin free tier only
  goes back to 2020, so the actual window is 2020-2025 (~5 years). This is
  a stated, deliberate limitation to be written up honestly in the final
  report, not hidden.
- This is Project 1 of a 3-project portfolio (Project 2: stat arb via
  cointegration; Project 3: factor model with neutralization). Only
  Project 1 is in progress right now.

## Methodology requirements (non-negotiable, don't skip these)
1. Point-in-time correctness: only use a fundamental data point in a given
   month if its Publish Date has already passed by that month.
2. Survivorship-bias-free universe: use historical S&P 500 membership, not
   today's list applied retroactively.
3. Cross-sectional ranking: predict relative performance (percentile rank
   within each month's universe), not raw returns.
4. Validation: purged, embargoed walk-forward cross-validation only. Never
   random k-fold on time series data. Reference: Lopez de Prado, "Advances
   in Financial Machine Learning," Ch. 7.
5. Factor attribution: after training models, run Fama-MacBeth regression
   against known factors (market, size, value, momentum, quality) to check
   whether the ML signal adds anything beyond known effects.
6. Interpretability: SHAP analysis on the tree-based model.
7. Backtest realism: decile long-short portfolio, report Sharpe ratio,
   turnover, and returns after realistic transaction costs (10-20bps).
8. Honest limitations section is mandatory in the final write-up — data
   window, survivorship/point-in-time caveats, capacity constraints, regime
   dependence, data snooping risk.

## Current status
- [x] SimFin data pulled and verified (quarterly + annual income statements)
- [x] Step 1: Build monthly point-in-time universe table
- [ ] Step 2: Pull daily price data for universe
- [ ] Step 3: Build features (fundamental + technical)
- [ ] Step 4: Build label (forward 1-month cross-sectional rank)
- [ ] Step 5: Build walk-forward validation harness
- [ ] Step 6: Train models (elastic net, LightGBM, small neural net)
- [ ] Step 7: Fama-MacBeth factor attribution
- [ ] Step 8: SHAP analysis
- [ ] Step 9: Portfolio backtest with costs
- [ ] Step 10: Write-up

## Working style
- Build one thing at a time, validate it before moving to the next (e.g.
  validate the universe script with spot-checks before building features
  on top of it).
- Prefer clarity and correctness over cleverness — this needs to survive
  scrutiny from a technical reviewer, not just run without errors.
- Keep raw vendor data (SimFin CSVs) out of version control if their terms
  of use restrict redistribution — check this, and gitignore if needed.
- `scripts/` holds real pipeline code (each step's script, meant to be
  reproducible and reviewed). `scratch/` is for raw vendor data caches and
  genuinely throwaway/exploratory files, not pipeline steps.
