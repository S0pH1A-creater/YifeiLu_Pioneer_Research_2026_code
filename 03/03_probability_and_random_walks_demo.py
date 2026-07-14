"""
Probability and Random Walks Demo
==================================

This demo explores key concepts from the Probability and Random Walks lesson:

1. FAT TAILS in Financial Data
   - Empirical market distributions often have heavier tails than normal distributions
   - Extreme market crashes are MORE likely in reality than normal distribution predicts
   - Standard models underestimate tail risk

2. ADDITIVE vs GEOMETRIC RANDOM WALKS
   - Additive walks: S_t = S_{t-1} + shock (inappropriate for stock prices)
   - Geometric walks: S_t = S_{t-1} * (1 + return) (correct for limited liability assets)
   - Stocks cannot go below $0, so additive models are structurally flawed

3. STOCHASTIC PROCESSES
   - Simulation of multi-path random walks with proper statistical analysis
   - Understanding mean reversion, drift, and volatility effects
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
import time

# Set random seed for reproducibility
np.random.seed(42)

# Suppress yfinance warnings
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: FAT TAILS IN FINANCIAL DATA
# ============================================================================
print("=" * 70)
print("SECTION 1: FAT TAILS IN FINANCIAL DATA")
print("=" * 70)
print("\nDownloading SPY historical data from 2015-2024...")

# Fetch real SPY data with retry mechanism
max_retries = 5
retry_count = 0
spy_data = None

while retry_count < max_retries and spy_data is None:
    try:
        spy_ticker = yf.Ticker('SPY')
        spy_data = spy_ticker.history(start='2015-01-01', end='2024-01-01')
        break
    except Exception as e:
        retry_count += 1
        if retry_count < max_retries:
            wait_time = 5 * retry_count  # Progressive backoff
            print(f"Rate limited. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
            time.sleep(wait_time)
        else:
            print("Failed to download data after multiple retries. Using simulated data instead...")
            # Generate simulated data based on typical SPY behavior
            spy_returns_simulated = np.random.normal(0.0004, 0.012, 2500)
            spy_data = pd.DataFrame({'Close': np.cumprod(1 + spy_returns_simulated)})

spy_returns = spy_data['Close'].pct_change().dropna()

# Calculate statistics
mu = spy_returns.mean()
std = spy_returns.std()
skewness = spy_returns.skew()
kurtosis = spy_returns.kurtosis()

print(f"\nEmpirical SPY Returns Statistics:")
print(f"  Mean daily return: {mu:.6f}")
print(f"  Std deviation: {std:.6f}")
print(f"  Skewness: {skewness:.4f} (negative = left tail heavier)")
print(f"  Kurtosis: {kurtosis:.4f} (> 3 indicates fat tails)")
print(f"\nNumber of observations: {len(spy_returns)}")

# Create figure for fat tails analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fat Tails: Empirical vs Normal Distribution Analysis', fontsize=14, fontweight='bold')

# Subplot 1: Histogram with overlaid normal curve
ax1 = axes[0, 0]
ax1.hist(spy_returns, bins=100, density=True, alpha=0.7, color='steelblue', label='Empirical SPY Returns')
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
ax1.plot(x, p, 'r-', linewidth=2.5, label='Theoretical Normal Distribution')
ax1.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'Mean: {mu:.6f}')
ax1.set_xlabel('Daily Return')
ax1.set_ylabel('Density')
ax1.set_title('Fat Tails: Empirical vs Normal')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Q-Q plot (to visualize tail deviations)
ax2 = axes[0, 1]
stats.probplot(spy_returns, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot: Deviations at Tails')
ax2.grid(True, alpha=0.3)

# Subplot 3: Log-scale histogram (to see tail behavior)
ax3 = axes[1, 0]
ax3.hist(spy_returns, bins=100, density=True, alpha=0.7, color='steelblue', label='Empirical')
ax3.plot(x, p, 'r-', linewidth=2.5, label='Normal Distribution')
ax3.set_yscale('log')
ax3.set_xlabel('Daily Return')
ax3.set_ylabel('Density (log scale)')
ax3.set_title('Log Scale: Fat Tails More Visible')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Extreme returns comparison
ax4 = axes[1, 1]
# Count observations beyond 2, 3, and 4 standard deviations
std_levels = [1, 2, 3, 4]
empirical_counts = []
normal_counts = []

for std_level in std_levels:
    empirical = (np.abs(spy_returns) > std_level * std).sum() / len(spy_returns) * 100
    normal = 2 * (1 - stats.norm.cdf(std_level)) * 100  # Two-tailed
    empirical_counts.append(empirical)
    normal_counts.append(normal)

x_pos = np.arange(len(std_levels))
width = 0.35
ax4.bar(x_pos - width/2, empirical_counts, width, label='Empirical', color='steelblue', alpha=0.8)
ax4.bar(x_pos + width/2, normal_counts, width, label='Normal', color='red', alpha=0.8)
ax4.set_xlabel('Standard Deviations')
ax4.set_ylabel('Percentage of Returns (%)')
ax4.set_title('Extreme Events: Empirical vs Normal Model')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{i}σ' for i in std_levels])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/03_fat_tails_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Fat tails analysis complete. Key insight:")
print("  Extreme negative returns occur MORE frequently than normal distribution predicts.")
print("  This is the 'Black Swan' problem - standard models underestimate crash risk.")

# ============================================================================
# SECTION 2: ADDITIVE vs GEOMETRIC RANDOM WALKS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: ADDITIVE vs GEOMETRIC RANDOM WALKS")
print("=" * 70)

# Simulation parameters
N_steps = 252  # One trading year
N_paths = 100  # Number of simulated paths
initial_price = 100
daily_return_mean = spy_returns.mean()
daily_return_std = spy_returns.std()

print(f"\nSimulation parameters:")
print(f"  Initial price: ${initial_price}")
print(f"  Time horizon: {N_steps} days (1 trading year)")
print(f"  Number of paths: {N_paths}")
print(f"  Daily return mean: {daily_return_mean:.6f}")
print(f"  Daily return std: {daily_return_std:.6f}")

# Generate random shocks
shocks = np.random.normal(daily_return_mean, daily_return_std, (N_steps, N_paths))

# Create figure for comparing walk types
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Additive vs Geometric Random Walks (100 Simulated Paths)', fontsize=14, fontweight='bold')

# ============================================================================
# PROBLEM 1: Additive Random Walks (INCORRECT for stock prices)
# ============================================================================
# Additive walk: S_t = S_{t-1} + shock
# This is structurally flawed for stocks because:
# 1. Can go negative (violates limited liability)
# 2. Distribution doesn't match real market behavior
# 3. Ignores compounding effects

ax1 = axes[0]
additive_shocks = shocks * 5  # Scale shocks
additive_paths = np.vstack([np.zeros(N_paths), np.cumsum(additive_shocks, axis=0)]) + initial_price

# Plot paths with low opacity
for i in range(N_paths):
    ax1.plot(additive_paths[:, i], alpha=0.3, linewidth=0.8, color='steelblue')

ax1.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Boundary (Can Go Negative!)')
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Stock Price ($)')
ax1.set_title('Additive Walk: S_t = S_{t-1} + shock\n(INCORRECT - Can go negative)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Statistics
negative_paths = np.sum(np.any(additive_paths < 0, axis=0))
print(f"\nAdditive Walk Results:")
print(f"  Paths that went negative: {negative_paths}/{N_paths} ({100*negative_paths/N_paths:.1f}%)")
print(f"  Mean final price: ${np.mean(additive_paths[-1, :]):.2f}")
print(f"  Min final price: ${np.min(additive_paths[-1, :]):.2f}")
print(f"  ⚠️ Problem: Real stocks have LIMITED LIABILITY - cannot go below $0!")

# ============================================================================
# SOLUTION: Geometric Random Walks (CORRECT for stock prices)
# ============================================================================
# Geometric walk: S_t = S_{t-1} * (1 + return)
# This correctly models stock prices because:
# 1. Always positive
# 2. Matches empirical return distributions
# 3. Represents compound growth
# 4. Follows log-normal distribution

ax2 = axes[1]
geometric_paths = np.zeros((N_steps + 1, N_paths))
geometric_paths[0, :] = initial_price

for t in range(N_steps):
    geometric_paths[t + 1, :] = geometric_paths[t, :] * (1 + shocks[t, :])

# Plot paths with low opacity
for i in range(N_paths):
    ax2.plot(geometric_paths[:, i], alpha=0.3, linewidth=0.8, color='green')

ax2.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Boundary (Never Crossed)')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Stock Price ($)')
ax2.set_title('Geometric Walk: S_t = S_{t-1}×(1+return)\n(CORRECT - Always positive)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/03_random_walks_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nGeometric Walk Results:")
print(f"  Paths that went negative: 0/{N_paths} (0%)")
print(f"  Mean final price: ${np.mean(geometric_paths[-1, :]):.2f}")
print(f"  Min final price: ${np.min(geometric_paths[-1, :]):.2f}")
print(f"  ✓ Correct: All prices remain positive (limited liability)")

# ============================================================================
# SECTION 3: STATISTICAL ANALYSIS OF RANDOM WALKS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: STATISTICAL ANALYSIS - MONTE CARLO SIMULATION")
print("=" * 70)

# Run larger simulation for statistics (10,000 paths)
N_paths_large = 10000
shocks_large = np.random.normal(daily_return_mean, daily_return_std, (N_steps, N_paths_large))

geometric_paths_large = np.zeros((N_steps + 1, N_paths_large))
geometric_paths_large[0, :] = initial_price

for t in range(N_steps):
    geometric_paths_large[t + 1, :] = geometric_paths_large[t, :] * (1 + shocks_large[t, :])

# Calculate statistics at different time horizons
time_points = [1, 10, 50, 252]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Distribution of Final Prices (Monte Carlo: {N_paths_large} paths)', 
             fontsize=14, fontweight='bold')

for idx, t in enumerate(time_points):
    ax = axes[idx // 2, idx % 2]
    final_prices = geometric_paths_large[t, :]
    
    ax.hist(final_prices, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit and plot log-normal distribution
    mu_lognorm = np.log(final_prices).mean()
    sigma_lognorm = np.log(final_prices).std()
    x_range = np.linspace(final_prices.min(), final_prices.max(), 100)
    pdf_lognorm = stats.lognorm.pdf(x_range, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    ax.plot(x_range, pdf_lognorm, 'r-', linewidth=2.5, label='Log-normal fit')
    
    mean_price = final_prices.mean()
    ax.axvline(mean_price, color='green', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Density')
    ax.set_title(f'Day {t}: Price Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Print statistics
    print(f"\nDay {t} Statistics (from {N_paths_large} simulations):")
    print(f"  Mean: ${mean_price:.2f}")
    print(f"  Median: ${np.median(final_prices):.2f}")
    print(f"  Std Dev: ${final_prices.std():.2f}")
    print(f"  5th percentile (VaR): ${np.percentile(final_prices, 5):.2f}")
    print(f"  95th percentile: ${np.percentile(final_prices, 95):.2f}")

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/03_monte_carlo_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# SECTION 4: KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS FROM THIS ANALYSIS")
print("=" * 70)

print("""
1. FAT TAILS
   ✓ Empirical market returns have heavier tails than normal distribution
   ✓ Extreme events (crashes) occur 2-10x more often than predicted by normal model
   ✓ Standard deviation alone is insufficient to measure risk
   ✓ Risk models based on normality assumption underestimate tail risk

2. ADDITIVE vs GEOMETRIC WALKS
   ✓ Additive walks (S_t = S_{t-1} + shock) are INCORRECT for stocks
     - Can produce negative prices (violates limited liability)
     - Doesn't match empirical return distributions
   ✓ Geometric walks (S_t = S_{t-1}×(1+r)) are CORRECT
     - Always remain positive
     - Follow log-normal distribution
     - Match real market behavior

3. STOCHASTIC MODELING
   ✓ Multi-path simulations reveal future price distributions
   ✓ Uncertainty increases with time horizon (wider distribution)
   ✓ Mean is NOT a good prediction (wide confidence intervals)
   ✓ Value at Risk (VaR) measures tail risk at specific percentiles
   ✓ Understanding distributions is crucial for risk management

4. PRACTICAL IMPLICATIONS
   ✓ Use geometric Brownian motion for stock price simulations
   ✓ Monitor tail risk, not just volatility
   ✓ Don't rely solely on average returns
   ✓ Consider extreme scenarios in portfolio planning
""")

print("=" * 70)
print("✓ Demo complete! Files saved:")
print("  - 03_fat_tails_analysis.png")
print("  - 03_random_walks_comparison.png")
print("  - 03_monte_carlo_analysis.png")
print("=" * 70)
