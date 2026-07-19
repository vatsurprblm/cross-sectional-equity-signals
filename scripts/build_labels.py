"""
Step 4: build the forward-return label with a 2-business-day execution lag.

Label = cross-sectional percentile rank (within each month, 0=worst,
1=best) of a stock's return from 2 trading days after this month's signal
date to 2 trading days after next month's signal date.

Why the 2-day lag: a signal computed as of month-end can't actually be
traded at that exact closing price — by the time you observe the signal
and place an order, some time has passed. Using the very next day's close
for both signal and execution (0-day lag) would be unrealistic and would
implicitly assume perfect, instant execution. The 2-day lag is a simple,
conservative stand-in for that real-world delay, and using the SAME 2-day
lag both for where the return period starts and where it ends keeps every
period exactly one calendar month long (not off by a couple of days).

Execution dates are offset in trading days (not calendar days) from the
same month-end trading-day calendar Step 3 uses for signal_date, so a
"2-day lag" means 2 actual market trading days, not 2 calendar days that
might span a weekend differently for different months.

Output: data/monthly_labels.csv (month, ticker, execution_date_start,
execution_date_end, forward_return, label)
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_features import load_universe, load_prices, build_month_signal_dates, UNIVERSE_TO_SIMFIN_TICKER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXECUTION_LAG_TRADING_DAYS = 2


def build_month_execution_dates(prices, month_signal_dates):
    """For each month, the trading day EXECUTION_LAG_TRADING_DAYS after
    that month's signal_date, using the same global trading-day calendar
    build_month_signal_dates() is built from."""
    all_dates = pd.Series(sorted(prices['date'].unique()))
    signal_idx = all_dates.searchsorted(month_signal_dates['signal_date'].values)
    exec_idx = signal_idx + EXECUTION_LAG_TRADING_DAYS
    execution_dates = month_signal_dates.copy()
    execution_dates['execution_date'] = [
        all_dates[i] if i < len(all_dates) else pd.NaT for i in exec_idx
    ]
    return execution_dates[['month', 'execution_date']]


def asof_execution_prices(prices, month_execution_dates):
    """Price as of each month's execution_date, per ticker (as-of merge,
    same backward + small-tolerance convention as Step 3)."""
    frames = []
    for ticker, g in prices.groupby('ticker', sort=False):
        g = g[['date', 'close']].sort_values('date')
        merged = pd.merge_asof(
            month_execution_dates.dropna(subset=['execution_date']).sort_values('execution_date'),
            g, left_on='execution_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=5),
        )
        merged['ticker'] = ticker
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True).drop(columns=['date'])
    return out.rename(columns={'close': 'execution_price'})


def compute_forward_return(execution_prices):
    """Forward return for month M = execution_price[M+1] / execution_price[M] - 1,
    via a pivot + shift(-1) so a genuinely missing next month gives NaN
    rather than silently pairing with the wrong row."""
    pivot = execution_prices.pivot(index='month', columns='ticker', values='execution_price').sort_index()
    fwd = pivot.shift(-1) / pivot - 1
    fwd = fwd.stack().rename('forward_return').reset_index()
    return fwd


def main():
    print("Loading universe and prices...")
    universe = load_universe()
    prices = load_prices()

    print("\nBuilding signal-date and execution-date calendars...")
    month_signal_dates = build_month_signal_dates(prices)
    month_execution_dates = build_month_execution_dates(prices, month_signal_dates)
    print(month_execution_dates.tail(3).to_string())

    print("\nComputing execution-date prices and forward returns...")
    execution_prices = asof_execution_prices(prices, month_execution_dates)
    forward_return = compute_forward_return(execution_prices)

    print("\nAssembling universe x label table...")
    universe['month'] = universe['month'].astype(str)
    forward_return['month'] = forward_return['month'].astype(str)
    out = universe[['month', 'ticker']].merge(forward_return, on=['month', 'ticker'], how='left')

    exec_dates_by_month = month_execution_dates.copy()
    exec_dates_by_month['month'] = exec_dates_by_month['month'].astype(str)
    exec_dates_by_month = exec_dates_by_month.rename(columns={'execution_date': 'execution_date_start'})
    out = out.merge(exec_dates_by_month, on='month', how='left')

    next_month_exec = exec_dates_by_month.copy()
    next_month_exec['month'] = (pd.PeriodIndex(next_month_exec['month'], freq='M') - 1).astype(str)
    next_month_exec = next_month_exec.rename(columns={'execution_date_start': 'execution_date_end'})
    out = out.merge(next_month_exec, on='month', how='left')

    print("\nComputing cross-sectional percentile rank label (0=worst, 1=best)...")
    out['label'] = out.groupby('month')['forward_return'].rank(pct=True)

    out = out.sort_values(['month', 'ticker']).reset_index(drop=True)
    out_path = os.path.join(PROJECT_ROOT, 'data', 'monthly_labels.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out):,} rows to {out_path}")

    return out


if __name__ == '__main__':
    main()
