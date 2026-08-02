"""Quick tests derived from `01_Python_and_Data_Basics.ipynb`.
Uses only the notebook's logic: forward-fill cleaning, normalization (divide by initial),
and rolling SMA computation. Creates synthetic closing prices based on the notebook's tickers
and runs simple checks.
"""

import pandas as pd

# Synthetic sample data (uses the same tickers and date range concept from the notebook)
dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
prices = {
    'AAPL': [130, 131, 129, 132, 134, 133, 135, 136, 138, 137],
    'MSFT': [220, 222, 221, 223, 225, 224, 226, 227, 230, 229],
    'SPY':  [380, 381, 379, 382, 384, 383, 385, 386, 388, 387],
}

data = pd.DataFrame(prices, index=dates)

# 1) Data cleaning (ffill + dropna) — from the notebook
cleaned = data.ffill().dropna()

# 2) Normalization (divide by initial price) — from the notebook
def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    return df / df.iloc[0]

normalized = normalize_prices(cleaned)

# 3) SMA function (rolling mean)
def sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window).mean()

# For demonstration with short sample data, use a smaller window
aapl_sma_3 = sma(cleaned['AAPL'], window=3)

# Simple checks/tests using only the file's information
def run_checks():
    results = {}

    # Check that forward-fill + dropna didn't introduce NaNs for this synthetic data
    results['missing_after_cleaning'] = int(cleaned.isna().sum().sum())

    # Normalization: first row should be 1.0 for all columns
    first_row_all_one = (normalized.iloc[0] == 1.0).all()
    results['normalization_first_row_is_one'] = bool(first_row_all_one)

    # SMA: for window=3, the third value should be the mean of first 3 AAPL prices
    expected_third = sum(prices['AAPL'][:3]) / 3
    sma_third = float(aapl_sma_3.iloc[2])
    results['sma_third_matches_expected'] = abs(sma_third - expected_third) < 1e-8

    return results

if __name__ == '__main__':
    print('\nSample data (first 5 rows):')
    print(data.head())

    print('\nCleaned data (first 5 rows):')
    print(cleaned.head())

    print('\nNormalized data (first 5 rows):')
    print(normalized.head())

    print('\nAAPL SMA (window=3, first 6 rows):')
    print(aapl_sma_3.head(6))

    checks = run_checks()
    print('\nChecks:')
    for k, v in checks.items():
        print(f"- {k}: {v}")
