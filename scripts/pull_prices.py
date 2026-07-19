"""
Pull daily OHLCV price data for every ticker in the monthly universe.

Date range starts a year before the universe window (2020-01) so that
Step 3 has enough trailing history to compute 12-month momentum and other
lookback features for the earliest universe months, not just from 2021.
It ends a few trading days after the universe window's last month
(2025-06) so Step 4's 2-day-execution-lag label can be computed for that
last month too — without this buffer, the label's forward-return window
would run past the end of the price data and come back all-NaN.

Prices are split/dividend-adjusted (auto_adjust=True) so returns aren't
distorted by artificial jumps at split/dividend events.

Output: data/daily_prices.csv (date, ticker, open, high, low, close, volume)
"""

import os
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

START_DATE = '2019-01-01'
END_DATE = '2025-07-10'

# Yahoo Finance uses a hyphen for share-class tickers where the S&P/SimFin
# source uses a dot (e.g. BRK.B). Only BRK.B appears in our universe file;
# all other dot-notation tickers were already excluded in build_universe.py.
UNIVERSE_TO_YFINANCE_TICKER = {
    'BRK.B': 'BRK-B',
}


def load_universe_tickers():
    universe_path = os.path.join(PROJECT_ROOT, 'data', 'monthly_universe.csv')
    universe = pd.read_csv(universe_path)
    return sorted(universe['ticker'].unique())


def main():
    tickers = load_universe_tickers()
    print(f"Universe tickers: {len(tickers)}")

    yf_tickers = [UNIVERSE_TO_YFINANCE_TICKER.get(t, t) for t in tickers]
    yf_to_universe = {UNIVERSE_TO_YFINANCE_TICKER.get(t, t): t for t in tickers}

    print(f"Downloading {START_DATE} to {END_DATE} via yfinance...")
    raw = yf.download(
        yf_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by='ticker',
        threads=True,
        progress=False,
    )

    records = []
    failed_or_empty = []

    for yf_ticker in yf_tickers:
        universe_ticker = yf_to_universe[yf_ticker]
        try:
            df = raw[yf_ticker].copy()
        except KeyError:
            failed_or_empty.append(universe_ticker)
            continue

        df = df.dropna(subset=['Close'])
        if df.empty:
            failed_or_empty.append(universe_ticker)
            continue

        df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['ticker'] = universe_ticker
        records.append(df)

    prices = pd.concat(records, ignore_index=True)
    prices = prices[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

    out_path = os.path.join(PROJECT_ROOT, 'data', 'daily_prices.csv')
    prices.to_csv(out_path, index=False)

    print(f"\nSaved {len(prices):,} rows for {prices['ticker'].nunique()} tickers to {out_path}")

    # Flag tickers with suspiciously short history relative to the median,
    # which usually means a mid-window delisting, late IPO, or symbol change.
    span = prices.groupby('ticker')['date'].agg(['min', 'max', 'count'])
    median_count = span['count'].median()
    short_history = span[span['count'] < 0.5 * median_count]

    # Record the coverage gaps as a reviewable artifact, not just console
    # output. Verified by hand: as of this pull, the "no_data" tickers are
    # almost all names that were acquired, merged, or taken private between
    # 2019-2025 (e.g. ABMD/J&J, TWTR/Musk buyout, XLNX/AMD, KSU/CP) — Yahoo
    # Finance tends to purge historical price data along with the live quote
    # once a ticker is delisted, not just stop updating it. A few (e.g. PKI,
    # HFC, WRK) are pure ticker renames where the company still trades under
    # a new symbol, which we are not chasing down for this pass (see chat/
    # project notes). "short_history" entries like KVUE are legitimate late
    # IPOs, not data errors.
    gap_records = [{'ticker': t, 'reason': 'no_data'} for t in failed_or_empty]
    gap_records += [
        {'ticker': t, 'reason': f'short_history ({int(row["count"])} rows, '
                                 f'{row["min"].date()} to {row["max"].date()})'}
        for t, row in short_history.iterrows()
    ]
    gaps_path = os.path.join(PROJECT_ROOT, 'data', 'price_data_gaps.csv')
    pd.DataFrame(gap_records).to_csv(gaps_path, index=False)
    print(f"Logged {len(gap_records)} coverage gaps to {gaps_path}")


if __name__ == '__main__':
    main()
