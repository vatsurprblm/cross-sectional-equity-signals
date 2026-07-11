"""
Build the monthly universe table:
- For each month from Jan 2020 to Jun 2025, identify S&P 500 constituents
  that also have data in the SimFin quarterly income dataset.
- Output: data/monthly_universe.csv  (month, ticker)
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A handful of S&P 500 constituents use a share-class ticker (dot notation,
# e.g. BRK.B) that doesn't exist verbatim in SimFin. Where the same company
# is covered under a different share class, map it explicitly rather than
# silently dropping it. Only BRK.B is resolvable this way — the other dot
# tickers in the S&P 500 source (AFS.A, AZA.A, BF.B, CIT.A, COC.B, FTL.A,
# GFS.A, LDW.B, RDS.A, TMC.A) have no SimFin coverage under any ticker.
SP500_TO_SIMFIN_TICKER = {
    'BRK.B': 'BRK-A',  # Berkshire Hathaway: S&P 500 tracks the B shares, SimFin only covers the A shares
}

# ── 1. Load S&P 500 membership data ─────────────────────────────────────
sp500_path = os.path.join(PROJECT_ROOT, 'data', 'sp500', 'sp500_ticker_start_end.csv')
sp500 = pd.read_csv(sp500_path)
sp500['start_date'] = pd.to_datetime(sp500['start_date'])
sp500['end_date'] = pd.to_datetime(sp500['end_date'])
sp500['simfin_ticker'] = sp500['ticker'].map(SP500_TO_SIMFIN_TICKER).fillna(sp500['ticker'])

print(f"S&P 500 ticker-start-end file: {len(sp500)} rows")
print(f"  Sample:\n{sp500.head()}\n")

# ── 2. Load SimFin quarterly income tickers ──────────────────────────────
simfin_path = os.path.join(PROJECT_ROOT, 'scratch', 'income_quarterly_all.csv')
simfin_q = pd.read_csv(simfin_path, usecols=['Ticker'])
simfin_tickers = set(simfin_q['Ticker'].unique())

print(f"SimFin quarterly data: {len(simfin_tickers)} unique tickers\n")

# ── 3. Build month range: Jan 2020 → Jun 2025 ───────────────────────────
months = pd.date_range('2020-01-01', '2025-06-01', freq='MS')  # month-start

# ── 4. Cross-reference: for each month, find S&P 500 members with SimFin data
records = []

for month_start in months:
    month_end = month_start + pd.offsets.MonthEnd(0)  # last day of month

    # Ticker was an S&P 500 member during this month if:
    #   start_date <= month_end  AND  (end_date >= month_start OR end_date is NaT)
    mask = (
        (sp500['start_date'] <= month_end) &
        (sp500['end_date'].isna() | (sp500['end_date'] >= month_start))
    )
    sp500_in_month = sp500.loc[mask, ['ticker', 'simfin_ticker']]

    # Keep the S&P 500 ticker as the output label, but match against SimFin
    # using simfin_ticker (identical for most rows, remapped for BRK.B etc.)
    overlap = sp500_in_month[sp500_in_month['simfin_ticker'].isin(simfin_tickers)]

    for ticker in sorted(overlap['ticker']):
        records.append({'month': month_start.strftime('%Y-%m'), 'ticker': ticker})

universe = pd.DataFrame(records)

# ── 5. Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(PROJECT_ROOT, 'data', 'monthly_universe.csv')
universe.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"  Total rows (month × ticker): {len(universe):,}")

# ── 6. Summary stats ────────────────────────────────────────────────────
unique_tickers = universe['ticker'].nunique()
counts_per_month = universe.groupby('month')['ticker'].count()

print(f"\n=== Summary ===")
print(f"  Unique companies across entire period: {unique_tickers}")
print(f"  Companies per month:")
print(f"    Min:    {counts_per_month.min()}")
print(f"    Max:    {counts_per_month.max()}")
print(f"    Mean:   {counts_per_month.mean():.1f}")
print(f"    Median: {counts_per_month.median():.1f}")
print(f"\n  Monthly breakdown:")
print(counts_per_month.to_string())
