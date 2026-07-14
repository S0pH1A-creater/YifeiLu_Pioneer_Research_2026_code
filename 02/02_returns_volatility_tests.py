"""Quick tests derived from `02_Returns_and_Volatility.ipynb`.
Uses only the notebook's logic: simple vs log returns, rolling volatility,
and stock vs index volatility comparison. Uses synthetic data based on realistic patterns.
"""

import pandas as pd
import numpy as np

# Synthetic price data for AAPL and SPY (simulates 4 years of daily prices)
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")

# Create realistic price series with drift and volatility
aapl_returns = np.random.normal(0.0005, 0.02, 1000)  # mean daily return, std dev
aapl_prices = 100 * np.exp(np.cumsum(aapl_returns))

spy_returns = np.random.normal(0.0003, 0.012, 1000)  # SPY: lower volatility
spy_prices = 300 * np.exp(np.cumsum(spy_returns))

aapl_data = pd.Series(aapl_prices, index=dates)
spy_data = pd.Series(spy_prices, index=dates)

# 1) Simple vs Log Returns (from notebook)
def simple_returns(prices: pd.Series) -> pd.Series:
    """Simple return: (S_t - S_{t-1}) / S_{t-1}"""
    return prices.pct_change().dropna()

def log_returns(prices: pd.Series) -> pd.Series:
    """Log return: ln(S_t / S_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()

aapl_simple = simple_returns(aapl_data)
aapl_log = log_returns(aapl_data)

# 2) Rolling Volatility (from notebook)
def rolling_volatility(returns: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    """Compute rolling volatility (annualized if requested)"""
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)  # 252 trading days per year
    return vol

aapl_vol = rolling_volatility(aapl_log, window=30, annualize=True)
spy_vol = rolling_volatility(log_returns(spy_data), window=30, annualize=True)

# 3) Comparisons for testing
def run_checks():
    results = {}

    # Check 1: Simple and log returns should be close for small changes
    mean_diff = abs(aapl_simple.mean() - aapl_log.mean())
    results['simple_log_means_close'] = mean_diff < 0.01

    # Check 2: Log returns should have additive property (sum ≈ log(price_end/price_start))
    log_sum = aapl_log.sum()
    log_ratio = np.log(aapl_data.iloc[-1] / aapl_data.iloc[0])
    results['log_returns_additive'] = abs(log_sum - log_ratio) < 0.01

    # Check 3: SPY volatility should be lower than AAPL on average
    aapl_avg_vol = aapl_vol.mean()
    spy_avg_vol = spy_vol.mean()
    results['spy_vol_lower_than_aapl'] = spy_avg_vol < aapl_avg_vol

    # Check 4: Rolling volatility should not have NaNs after 30 days
    first_valid = aapl_vol.first_valid_index()
    days_until_valid = (first_valid - aapl_vol.index[0]).days
    results['rolling_vol_valid_after_window'] = days_until_valid == 29  # 30-day window

    return results

if __name__ == '__main__':
    print('\n=== Simple vs Log Returns ===')
    print(f'Mean Simple Return (AAPL): {aapl_simple.mean():.6f}')
    print(f'Mean Log Return (AAPL):    {aapl_log.mean():.6f}')
    print(f'First 5 simple returns:\n{aapl_simple.head()}')
    print(f'First 5 log returns:\n{aapl_log.head()}')

    print('\n=== Rolling Volatility ===')
    print(f'Mean Rolling Volatility (AAPL): {aapl_vol.mean():.6f}')
    print(f'Mean Rolling Volatility (SPY):  {spy_vol.mean():.6f}')
    print(f'Max Volatility (AAPL): {aapl_vol.max():.6f}')
    print(f'Max Volatility (SPY):  {spy_vol.max():.6f}')
    print(f'Volatility (last 5 days):\n{aapl_vol.tail()}')

    print('\n=== Return Distributions ===')
    print(f'AAPL Returns - Mean: {aapl_log.mean():.6f}, Std: {aapl_log.std():.6f}')
    print(f'SPY Returns  - Mean: {log_returns(spy_data).mean():.6f}, Std: {log_returns(spy_data).std():.6f}')

    print('\n=== Tests ===')
    checks = run_checks()
    for k, v in checks.items():
        status = '✓' if v else '✗'
        print(f'{status} {k}: {v}')
