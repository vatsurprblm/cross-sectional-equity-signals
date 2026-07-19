"""
Step 5: purged, embargoed, expanding-window walk-forward validation harness.

Why this is needed (plain language): a normal random k-fold split (common
in ML) shuffles rows randomly into train/test buckets. For time series
data that's actively wrong — it lets the model train on data from AFTER
the period it's tested on, which is a form of look-ahead bias called
"leakage." Walk-forward validation instead always trains on the past and
tests on a later period, which mimics how the model would actually be used
in real trading (you only ever have data up to today).

"Purged + embargoed" adds one more layer on top of a plain walk-forward
split. Our label (Step 4) looks 1 month into the future — a training
sample from month T has a label built from price data that stretches into
month T+1. If the test set started right at month T+1, that overlap would
leak information about the test period into the training set. The 1-month
embargo gap between train and test removes exactly that overlap: it's not
tested on and not trained on, just skipped, so training data and test data
never share any underlying price information. This also functions as the
"purge" Lopez de Prado describes — since our label horizon is exactly 1
month, a 1-month embargo is sufficient to purge all overlap; a longer
label horizon would need a longer embargo.

Design (matches CLAUDE.md's stated spec, reconciling "test T+1 to T+3" with
"1-month embargo between train end and test start" — the coherent reading
is the embargo pushes the test window out by one month):
  - Expanding window: fold i's training set is every month from the very
    start up to that fold's train_end (not a fixed-size rolling window —
    more data is always better once it's legitimately in the past).
  - First fold's train_end is exactly the 24-month minimum.
  - Between train_end and test_start there is exactly 1 embargoed month,
    used in neither train nor test.
  - Each test window is 3 months, and test windows across folds are
    contiguous and non-overlapping (fold i+1's train_end = fold i's
    test_end, so the next fold's expanded training set absorbs exactly the
    months just tested, and nothing is tested twice).
  - The final fold is kept even if fewer than 3 test months remain
    (truncated to whatever's left), so no available data goes unused —
    documented explicitly rather than silently dropped.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_TRAIN_MONTHS = 24
EMBARGO_MONTHS = 1
TEST_MONTHS = 3


def generate_folds(months, min_train=MIN_TRAIN_MONTHS, embargo=EMBARGO_MONTHS, test_size=TEST_MONTHS):
    """months: sorted list of all available month strings (e.g. '2020-01').
    Returns a list of dicts: {train, embargo, test} each a list of month
    strings, in chronological order."""
    months = sorted(months)
    n = len(months)
    folds = []
    train_end_idx = min_train - 1  # 0-indexed position of the last training month

    while True:
        embargo_start_idx = train_end_idx + 1
        embargo_end_idx = embargo_start_idx + embargo - 1
        test_start_idx = embargo_end_idx + 1
        if test_start_idx >= n:
            break
        test_end_idx = min(test_start_idx + test_size - 1, n - 1)

        folds.append({
            'train': months[0:train_end_idx + 1],
            'embargo': months[embargo_start_idx:embargo_end_idx + 1],
            'test': months[test_start_idx:test_end_idx + 1],
        })

        train_end_idx = test_end_idx  # expand: next fold's train absorbs this fold's test

    return folds


def validate_folds(folds, months):
    """Structural checks: disjointness, embargo size/position, expanding
    window property, contiguous non-overlapping test windows, minimum
    training window. Raises AssertionError on any violation."""
    months = sorted(months)
    prev_test_end = None

    for i, fold in enumerate(folds):
        train, embargo, test = fold['train'], fold['embargo'], fold['test']

        assert set(train).isdisjoint(embargo), f"fold {i}: train/embargo overlap"
        assert set(train).isdisjoint(test), f"fold {i}: train/test overlap"
        assert set(embargo).isdisjoint(test), f"fold {i}: embargo/test overlap"

        assert len(embargo) == EMBARGO_MONTHS, f"fold {i}: embargo size wrong"
        assert months.index(embargo[0]) == months.index(train[-1]) + 1, \
            f"fold {i}: embargo doesn't immediately follow train"
        assert months.index(test[0]) == months.index(embargo[-1]) + 1, \
            f"fold {i}: test doesn't immediately follow embargo"

        if i == 0:
            assert len(train) == MIN_TRAIN_MONTHS, "first fold's training window isn't the minimum"
        else:
            assert train[-1] == prev_test_end, \
                f"fold {i}: training window didn't expand to absorb prior fold's test months"

        prev_test_end = test[-1]

    print(f"All {len(folds)} folds pass structural validation "
          f"(no train/embargo/test overlap, correct embargo size and position, "
          f"expanding window, minimum {MIN_TRAIN_MONTHS}-month first training set).")


def main():
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'monthly_labels.csv'))
    months = sorted(labels['month'].unique())

    folds = generate_folds(months)
    validate_folds(folds, months)

    print(f"\n{len(months)} total months available ({months[0]} to {months[-1]}).")
    print(f"{len(folds)} folds generated structurally.\n")

    print(f"{'fold':>4} | {'train (n, range)':<28} | {'embargo':<9} | {'test (range)':<17} | test rows w/ label")
    label_counts = labels.groupby('month')['label'].apply(lambda x: x.notna().sum())
    dropped = []
    kept_folds = []
    for i, fold in enumerate(folds):
        train, embargo, test = fold['train'], fold['embargo'], fold['test']
        test_labeled_rows = sum(label_counts.get(m, 0) for m in test)
        print(f"{i:>4} | {len(train):>3} months, {train[0]}..{train[-1]:<10} | "
              f"{embargo[0]:<9} | {test[0]}..{test[-1]:<10} | {test_labeled_rows}")
        if test_labeled_rows == 0:
            dropped.append(i)
        else:
            kept_folds.append(fold)

    if dropped:
        # The last universe month (2025-06) has no forward-return label at
        # all (no "next month" exists to compute it from — see Step 4), so
        # any fold whose ENTIRE test window falls on such months is
        # structurally valid but useless for evaluation: dropped rather
        # than kept as a no-op fold with nothing to compute a metric on.
        print(f"\nDropped fold(s) {dropped} — zero usable test labels "
              f"(test window falls entirely on a month with no forward-return label).")

    print(f"\n{len(kept_folds)} usable folds for evaluation.")
    return kept_folds


if __name__ == '__main__':
    main()
