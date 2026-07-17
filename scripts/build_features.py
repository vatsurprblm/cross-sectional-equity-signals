"""
Step 3: build the monthly, point-in-time, sector-neutralised feature panel.

Overview of the approach:
  1. Build a common "month -> signal date" calendar (the last trading day of
     each calendar month), shared across all tickers.
  2. Technical features:
     - mom_1 / mom_12_1 use a monthly close-price grid (ticker x month) so
       "1 month ago" / "13 months ago" means an actual calendar month, not
       an approximate number of trading days.
     - vol_20d / vol_60d / price_52w_high / volume_ratio are inherently
       trading-day-window stats, computed on the daily panel, then read off
       as of each month's signal date via merge_asof.
  3. Fundamental features: point-in-time as-of joins (Publish Date <= signal
     date) against quarterly income/balance data, after filtering the known
     Publish Date < Report Date data quality issue. Net income is put on a
     trailing-twelve-month (TTM) basis (sum of the last 4 published
     quarters) before computing earnings_yield/roe/earnings_growth_yoy,
     which is the standard convention (a single quarter's net income over a
     full-year market cap would understate earnings yield by ~4x and isn't
     comparable across companies with different fiscal calendars).
  4. Sector neutralisation: z-score each feature within (month, sector,
     market-cap quintile), except log_market_cap which is z-scored within
     (month, sector) only — neutralising size by its own quintile would
     gut the size signal by construction.

Output: data/monthly_features.csv
"""

import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Universe/price labels use dot notation (BRK.B); SimFin fundamentals use
# hyphen notation (BRK-A) for the same company. Same crosswalk as
# build_universe.py / pull_prices.py.
UNIVERSE_TO_SIMFIN_TICKER = {'BRK.B': 'BRK-A'}

# SimFin's "Shares (Basic)" for BRK-A is the A-share-equivalent total share
# count, but our price series for BRK.B is the B-share price. Berkshire's
# A/B conversion ratio has been fixed at 1:1500 since the 2010 stock split
# (1 A-share = 1500 B-shares), so multiplying A-equivalent shares by the
# B-share price understates market cap ~1500x. Convert the B-share price to
# an A-equivalent price before computing market cap for this ticker only.
SHARE_CLASS_PRICE_MULTIPLIER = {'BRK.B': 1500}

# Point-in-time fundamentals lookups are capped at this many days stale to
# avoid silently using ancient data for a ticker with a reporting gap.
MAX_FUNDAMENTALS_STALENESS_DAYS = 500

# Minimum members required in a (month, sector, cap-quintile) group before
# z-scoring it directly; smaller groups fall back to (month, sector) only,
# since a std computed on 1-2 names is unstable/meaningless.
MIN_NEUTRALISATION_GROUP_SIZE = 3


def load_universe():
    universe = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'monthly_universe.csv'))
    universe['simfin_ticker'] = universe['ticker'].map(UNIVERSE_TO_SIMFIN_TICKER).fillna(universe['ticker'])
    universe['month'] = pd.PeriodIndex(universe['month'], freq='M')
    return universe


def load_prices():
    prices = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'daily_prices.csv'), parse_dates=['date'])
    prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)
    return prices


def build_month_signal_dates(prices):
    """One row per calendar month: the last trading day <= that month's end,
    based on the full market's trading calendar (union of all tickers' dates)."""
    all_dates = pd.Series(sorted(prices['date'].unique()))
    months = pd.period_range(all_dates.min().to_period('M'), all_dates.max().to_period('M'), freq='M')
    month_end_calendar = months.to_timestamp(how='end').normalize()
    idx = all_dates.searchsorted(month_end_calendar, side='right') - 1
    signal_dates = pd.DataFrame({
        'month': months,
        'signal_date': [all_dates[i] if i >= 0 else pd.NaT for i in idx],
    })
    return signal_dates


def build_monthly_close_grid(prices, month_signal_dates):
    """Ticker x month grid of close price as of each month's signal date,
    via a per-ticker as-of merge (tolerance a few days, to tolerate a ticker
    not trading on the exact common signal date)."""
    frames = []
    for ticker, g in prices.groupby('ticker', sort=False):
        g = g[['date', 'close', 'volume']].sort_values('date')
        merged = pd.merge_asof(
            month_signal_dates.sort_values('signal_date'),
            g, left_on='signal_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=5),
        )
        merged['ticker'] = ticker
        frames.append(merged)
    grid = pd.concat(frames, ignore_index=True)
    grid = grid.drop(columns=['date'])
    return grid


def compute_momentum_features(monthly_close_grid):
    """mom_1 and mom_12_1 via shifts on a properly-indexed ticker x month
    pivot, so a shift(13) genuinely means 13 calendar months, NaN if that
    month is missing rather than silently grabbing the wrong row."""
    pivot = monthly_close_grid.pivot(index='month', columns='ticker', values='close').sort_index()

    mom_1 = pivot / pivot.shift(1) - 1
    mom_12_1 = pivot.shift(1) / pivot.shift(13) - 1

    mom_1 = mom_1.stack().rename('mom_1').reset_index()
    mom_12_1 = mom_12_1.stack().rename('mom_12_1').reset_index()
    out = mom_1.merge(mom_12_1, on=['month', 'ticker'], how='outer')
    return out


def compute_daily_technical_panel(prices):
    """Trading-day-window technical features computed on the full daily
    panel, one row per (ticker, date)."""
    prices = prices.sort_values(['ticker', 'date']).copy()
    g = prices.groupby('ticker')

    daily_return = g['close'].pct_change()
    prices['vol_20d'] = daily_return.groupby(prices['ticker']).transform(
        lambda x: x.rolling(20).std() * np.sqrt(252))
    prices['vol_60d'] = daily_return.groupby(prices['ticker']).transform(
        lambda x: x.rolling(60).std() * np.sqrt(252))
    prices['high_252d'] = g['close'].transform(lambda x: x.rolling(252, min_periods=100).max())
    prices['price_52w_high'] = prices['close'] / prices['high_252d']
    vol_20d_avg = g['volume'].transform(lambda x: x.rolling(20).mean())
    vol_60d_avg = g['volume'].transform(lambda x: x.rolling(60).mean())
    prices['volume_ratio'] = vol_20d_avg / vol_60d_avg

    return prices[['ticker', 'date', 'vol_20d', 'vol_60d', 'price_52w_high', 'volume_ratio']]


def asof_technical_features(daily_technical_panel, month_signal_dates):
    frames = []
    for ticker, g in daily_technical_panel.groupby('ticker', sort=False):
        g = g.sort_values('date')
        merged = pd.merge_asof(
            month_signal_dates.sort_values('signal_date'),
            g, left_on='signal_date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=5),
        )
        merged['ticker'] = ticker
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True).drop(columns=['date'])
    return out


def load_fundamentals():
    """Load quarterly income + balance sheet, apply the crosswalk, and
    filter the known Publish Date < Report Date data quality issue."""
    income = pd.read_csv(os.path.join(PROJECT_ROOT, 'scratch', 'income_quarterly_all.csv'))
    balance = pd.read_csv(os.path.join(PROJECT_ROOT, 'scratch', 'balance_quarterly_all.csv'))

    for df in (income, balance):
        df['Report Date'] = pd.to_datetime(df['Report Date'])
        df['Publish Date'] = pd.to_datetime(df['Publish Date'])

    income_before = len(income)
    income = income[income['Publish Date'] >= income['Report Date']].copy()
    balance_before = len(balance)
    balance = balance[balance['Publish Date'] >= balance['Report Date']].copy()
    print(f"  Filtered income: {income_before - len(income)} bad Publish<Report rows dropped")
    print(f"  Filtered balance: {balance_before - len(balance)} bad Publish<Report rows dropped")

    balance = fix_shares_units_bug(balance)

    return income, balance


def fix_shares_units_bug(balance):
    """SimFin data quality bug: a handful of quarterly rows report Shares
    (Basic/Diluted) in millions instead of raw share count (e.g. AON's
    2022-09-30 quarter shows 210.0 instead of ~210,000,000; MCD's last 4
    available quarters show 715-718 instead of ~715-722 million). Detected
    by comparing each row to that ticker's own historical median share
    count — any row under 1% of its own median (only when that median is
    itself large enough to be a real security, not a penny stock) is
    assumed to be reported in millions and rescaled by 1e6."""
    balance = balance.sort_values(['Ticker', 'Report Date']).copy()
    for col in ['Shares (Basic)', 'Shares (Diluted)']:
        median = balance.groupby('Ticker')[col].transform('median')
        suspect = (balance[col] / median < 0.01) & balance[col].notna() & (median > 1_000_000)
        if suspect.any():
            print(f"  Fixed {suspect.sum()} rows with likely shares-in-millions units bug ({col})")
            balance.loc[suspect, col] = balance.loc[suspect, col] * 1_000_000
    return balance


def compute_ttm_net_income(income):
    """Trailing-twelve-month net income per ticker, as a point-in-time
    event series keyed by the Publish Date of the 4th (most recent) quarter
    in the window. Only computed when the 4 quarters are roughly
    consecutive (guards against gaps in reporting being silently summed as
    if consecutive). Summing 4 quarterly Report Dates ~91 days apart spans
    the *oldest to newest of the 4* by ~9 months (3 gaps), not 12 — the
    12-month coverage comes from the flows summed, not this date span."""
    income = income.sort_values(['Ticker', 'Report Date']).copy()
    g = income.groupby('Ticker')
    income['ttm_net_income'] = g['Net Income'].transform(lambda x: x.rolling(4).sum())
    income['quarters_span_days'] = g['Report Date'].transform(lambda x: (x - x.shift(3)).dt.days)
    valid_span = income['quarters_span_days'].between(240, 320)
    income.loc[~valid_span, 'ttm_net_income'] = np.nan
    return income[['Ticker', 'Publish Date', 'ttm_net_income']].dropna().rename(columns={'Ticker': 'ticker'})


def asof_fundamentals(events, universe, date_col, value_cols, lag_days=0):
    """As-of join: for each (ticker, month) in the universe, find the most
    recent `events` row with Publish Date <= (signal_date - lag_days),
    capped at MAX_FUNDAMENTALS_STALENESS_DAYS old."""
    lookup_dates = universe[['month', 'ticker', 'simfin_ticker', 'signal_date']].copy()
    lookup_dates['lookup_date'] = lookup_dates['signal_date'] - pd.Timedelta(days=lag_days)

    frames = []
    for simfin_ticker, g in lookup_dates.groupby('simfin_ticker', sort=False):
        ev = events[events['ticker'] == simfin_ticker].sort_values(date_col).drop(columns=['ticker'])
        if ev.empty:
            merged = g.copy()
            for c in value_cols:
                merged[c] = np.nan
            merged['fundamentals_publish_date'] = pd.NaT
        else:
            merged = pd.merge_asof(
                g.sort_values('lookup_date'), ev, left_on='lookup_date', right_on=date_col,
                direction='backward',
                tolerance=pd.Timedelta(days=MAX_FUNDAMENTALS_STALENESS_DAYS),
            )
            merged = merged.rename(columns={date_col: 'fundamentals_publish_date'})
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=['lookup_date'])


def compute_market_cap_quintile(df, month_col='month', value_col='market_cap'):
    def qcut_safe(x):
        try:
            return pd.qcut(x, 5, labels=False, duplicates='drop')
        except ValueError:
            return pd.Series(np.nan, index=x.index)
    return df.groupby(month_col)[value_col].transform(qcut_safe)


def sector_neutralise(df, feature_cols, group_cols):
    def zscore(x):
        std = x.std()
        return (x - x.mean()) / std if std and not np.isnan(std) and std > 1e-8 else np.nan

    for col in feature_cols:
        df[f'z_{col}'] = df.groupby(group_cols)[col].transform(zscore)

        # Fallback for (month, sector, cap-quintile) groups that are either
        # too small (std on 1-2 names is noise) OR have a missing cap
        # quintile (e.g. market cap unknown because shares outstanding is
        # missing — a fundamentals gap unrelated to this feature, most
        # commonly a technical feature that IS available and shouldn't lose
        # its neutralised value just because cap-quintile isn't known).
        # pandas groupby drops NaN keys, so group size comes back NaN (not
        # small) for those rows — NaN < N is always False, so it must be
        # checked explicitly rather than folded into the size comparison.
        if len(group_cols) == 3:
            fallback_group = group_cols[:2]
            group_sizes = df.groupby(group_cols)[col].transform('count')
            needs_fallback = (group_sizes < MIN_NEUTRALISATION_GROUP_SIZE) | group_sizes.isnull()
            can_fallback = df[col].notna() & df[group_cols[1]].notna()
            apply_mask = needs_fallback & can_fallback
            if apply_mask.any():
                fallback_z = df.groupby(fallback_group)[col].transform(zscore)
                df.loc[apply_mask, f'z_{col}'] = fallback_z[apply_mask]

    return df


def main():
    print("Loading universe, prices, fundamentals...")
    universe = load_universe()
    prices = load_prices()
    income, balance = load_fundamentals()

    print("\nBuilding month -> signal date calendar...")
    month_signal_dates = build_month_signal_dates(prices)
    print(f"  {len(month_signal_dates)} months, {month_signal_dates['signal_date'].isnull().sum()} unmapped")

    print("\nComputing monthly close-price grid and momentum features...")
    monthly_close_grid = build_monthly_close_grid(prices, month_signal_dates)
    momentum = compute_momentum_features(monthly_close_grid)

    print("\nComputing daily technical panel (vol_20d, vol_60d, price_52w_high, volume_ratio)...")
    daily_technical = compute_daily_technical_panel(prices)
    technical_asof = asof_technical_features(daily_technical, month_signal_dates)

    print("\nAssembling universe x signal_date base table...")
    base = universe.merge(month_signal_dates, on='month', how='left')

    base = base.merge(momentum, on=['month', 'ticker'], how='left')
    base = base.merge(
        technical_asof[['month', 'ticker', 'vol_20d', 'vol_60d', 'price_52w_high', 'volume_ratio']],
        on=['month', 'ticker'], how='left',
    )
    base = base.merge(
        monthly_close_grid[['month', 'ticker', 'close']].rename(columns={'close': 'price_at_signal'}),
        on=['month', 'ticker'], how='left',
    )

    print("\nComputing point-in-time fundamentals (TTM net income, book equity, shares)...")
    ttm_ni = compute_ttm_net_income(income)
    ttm_now = asof_fundamentals(ttm_ni, base, 'Publish Date', ['ttm_net_income'], lag_days=0)
    ttm_now = ttm_now.rename(columns={'ttm_net_income': 'ttm_net_income_now'})
    ttm_1y_ago = asof_fundamentals(ttm_ni, base, 'Publish Date', ['ttm_net_income'], lag_days=365)
    ttm_1y_ago = ttm_1y_ago.rename(columns={'ttm_net_income': 'ttm_net_income_1y_ago'})

    balance_events = balance.rename(columns={'Ticker': 'ticker'})
    equity_now = asof_fundamentals(
        balance_events, base, 'Publish Date',
        ['Total Equity', 'Shares (Basic)'], lag_days=0,
    )
    equity_now = equity_now.rename(columns={'Total Equity': 'book_equity', 'Shares (Basic)': 'shares_basic'})

    base = base.merge(
        ttm_now[['month', 'ticker', 'ttm_net_income_now', 'fundamentals_publish_date']],
        on=['month', 'ticker'], how='left',
    )
    base = base.merge(
        ttm_1y_ago[['month', 'ticker', 'ttm_net_income_1y_ago']],
        on=['month', 'ticker'], how='left',
    )
    base = base.merge(
        equity_now[['month', 'ticker', 'book_equity', 'shares_basic']],
        on=['month', 'ticker'], how='left',
    )

    print("\nComputing fundamental + combined features...")
    price_multiplier = base['ticker'].map(SHARE_CLASS_PRICE_MULTIPLIER).fillna(1)
    base['market_cap'] = base['shares_basic'] * base['price_at_signal'] * price_multiplier
    base['log_market_cap'] = np.log(base['market_cap'])
    base['earnings_yield'] = base['ttm_net_income_now'] / base['market_cap']
    base['roe'] = np.where(base['book_equity'] > 0, base['ttm_net_income_now'] / base['book_equity'], np.nan)
    base['earnings_growth_yoy'] = (
        (base['ttm_net_income_now'] - base['ttm_net_income_1y_ago']) / base['ttm_net_income_1y_ago'].abs()
    )
    base['earnings_momentum'] = base['earnings_growth_yoy'] * base['mom_12_1']

    print("\nLoading sector reference data...")
    companies = pd.read_csv(os.path.join(PROJECT_ROOT, 'scratch', 'companies_us.csv'))
    industries = pd.read_csv(os.path.join(PROJECT_ROOT, 'scratch', 'industries.csv'))
    sector_map = companies.merge(industries, on='IndustryId', how='left')[['Ticker', 'Sector']]
    sector_map = sector_map.rename(columns={'Ticker': 'simfin_ticker', 'Sector': 'sector'})
    base = base.merge(sector_map, on='simfin_ticker', how='left')

    print("\nComputing market cap quintiles and sector-neutralising features...")
    base['market_cap_quintile'] = compute_market_cap_quintile(base)

    feature_cols = [
        'mom_12_1', 'mom_1', 'vol_20d', 'vol_60d', 'price_52w_high', 'volume_ratio',
        'earnings_yield', 'roe', 'earnings_growth_yoy', 'earnings_momentum',
    ]
    base = sector_neutralise(base, feature_cols, group_cols=['month', 'sector', 'market_cap_quintile'])
    base = sector_neutralise(base, ['log_market_cap'], group_cols=['month', 'sector'])

    out_cols = (
        ['month', 'ticker', 'signal_date', 'sector', 'market_cap_quintile', 'market_cap',
         'fundamentals_publish_date']
        + feature_cols + ['log_market_cap']
        + [f'z_{c}' for c in feature_cols] + ['z_log_market_cap']
    )
    out = base[out_cols].sort_values(['month', 'ticker']).reset_index(drop=True)
    out['month'] = out['month'].astype(str)

    out_path = os.path.join(PROJECT_ROOT, 'data', 'monthly_features.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out):,} rows to {out_path}")

    return out


if __name__ == '__main__':
    main()
