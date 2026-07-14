"""
Monte Carlo Simulations Demo
=====================================
This demo simulates the concepts from 05_Monte_Carlo_Simulations.ipynb

Concepts covered:
1. Multiple Paths Simulation: Generate N stock price paths using GBM
2. The Cone of Uncertainty: Visualize expanding confidence intervals
3. Law of Large Numbers: Show convergence of estimates to true mean

We use real financial data from yfinance to estimate parameters (mu, sigma)
and then run Monte Carlo simulations on a real stock.
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

print("=" * 80)
print("MONTE CARLO SIMULATIONS DEMO")
print("=" * 80)

# ============================================================================
# STEP 1: Download Real Financial Data to Estimate Parameters
# ============================================================================
print("\n[STEP 1] Downloading real financial data...")

# Download 5 years of historical data for Apple stock
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Try to download data with retry logic
data = None
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Attempting to download {ticker} data (attempt {attempt+1}/{max_retries})...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            print(f"✓ Downloaded {len(data)} trading days of {ticker} data")
            break
    except Exception as e:
        print(f"✗ API attempt {attempt+1} failed: {type(e).__name__}")
        if attempt < max_retries - 1:
            wait_time = 5 * (attempt + 1)
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"  Using synthetic data as fallback...")

# If download failed or returned empty, use synthetic data
if data is None or len(data) == 0:
    print("Generating synthetic historical data that mimics real stock behavior...")
    # Create realistic synthetic data: 5 years of trading days (~1260 days)
    n_days = 1260
    drift = 0.0002  # daily drift (~5% annual)
    volatility = 0.015  # daily volatility (~24% annual)
    S0_init = 150  # Starting price
    
    returns_synthetic = np.random.normal(drift, volatility, n_days)
    close_prices = S0_init * np.exp(np.cumsum(returns_synthetic))
    
    # Create DataFrame to match yfinance structure
    dates = pd.date_range(end=end_date, periods=n_days, freq='B')
    data = pd.DataFrame({'Adj Close': close_prices}, index=dates)
    print(f"✓ Generated {len(data)} synthetic trading days")

# Calculate daily returns
if isinstance(data, dict):
    returns = np.diff(data['Adj Close']) / data['Adj Close'][:-1]
else:
    returns = data['Adj Close'].pct_change().dropna()

# Estimate parameters from historical data
mu_daily = returns.mean()
sigma_daily = returns.std()

# Annualize parameters (252 trading days per year)
mu = mu_daily * 252
sigma = sigma_daily * np.sqrt(252)

# Get current stock price
if isinstance(data, dict):
    S0 = data['Adj Close'][-1]
else:
    S0 = data['Adj Close'].iloc[-1]

print(f"Current Price (S0): ${S0:.2f}")
print(f"Annualized Drift (μ): {mu:.4f} ({mu*100:.2f}%)")
print(f"Annualized Volatility (σ): {sigma:.4f} ({sigma*100:.2f}%)")

# ============================================================================
# STEP 2: Multiple Paths Simulation
# ============================================================================
print("\n[STEP 2] Generating 100 Monte Carlo paths (1 year forward)...")

T = 1  # 1 year horizon
N_steps = 252  # daily steps
dt = T / N_steps
N_sims = 100

# Initialize paths array: (N_steps, N_sims)
paths = np.zeros((N_steps, N_sims))
paths[0] = S0

# Geometric Brownian Motion formula:
# S(t) = S(t-1) * exp((μ - σ²/2)*dt + σ*√dt*Z)
# where Z ~ N(0,1)
for t in range(1, N_steps):
    Z = np.random.standard_normal(N_sims)
    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Plot 100 paths
plt.figure(figsize=(12, 6))
plt.plot(paths, lw=0.5, alpha=0.6, color='steelblue')
plt.title(f'Monte Carlo Simulation: 100 Paths for {ticker} (1-Year Horizon)', fontsize=14, fontweight='bold')
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/05/01_multiple_paths.png', dpi=300)
print("✓ Saved: 01_multiple_paths.png")
plt.close()

# ============================================================================
# STEP 3: The Cone of Uncertainty
# ============================================================================
print("\n[STEP 3] Computing the Cone of Uncertainty with 10,000 simulations...")

N_sims_large = 10000
paths_large = np.zeros((N_steps, N_sims_large))
paths_large[0] = S0

# Generate 10,000 paths
for t in range(1, N_steps):
    Z = np.random.standard_normal(N_sims_large)
    paths_large[t] = paths_large[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Calculate percentiles
mean_path = np.mean(paths_large, axis=1)
median_path = np.median(paths_large, axis=1)
p_05 = np.percentile(paths_large, 5, axis=1)
p_25 = np.percentile(paths_large, 25, axis=1)
p_75 = np.percentile(paths_large, 75, axis=1)
p_95 = np.percentile(paths_large, 95, axis=1)

# Plot the Cone of Uncertainty
plt.figure(figsize=(12, 7))

# Fill multiple bands for better visualization
plt.fill_between(range(N_steps), p_05, p_95, color='red', alpha=0.15, label='90% Confidence Interval (5th-95th %ile)')
plt.fill_between(range(N_steps), p_25, p_75, color='orange', alpha=0.25, label='50% Confidence Interval (25th-75th %ile)')

# Plot central tendency lines
plt.plot(mean_path, color='darkred', linewidth=2.5, label='Mean Path', linestyle='-')
plt.plot(median_path, color='darkblue', linewidth=2.5, label='Median Path', linestyle='--')

# Plot boundary lines
plt.plot(p_05, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='5th Percentile')
plt.plot(p_95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95th Percentile')

# Add starting price reference line
plt.axhline(S0, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Current Price (${S0:.2f})')

plt.title(f'The Cone of Uncertainty for {ticker}: Stock Price Distribution Over Time', 
          fontsize=14, fontweight='bold')
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/05/02_cone_of_uncertainty.png', dpi=300)
print("✓ Saved: 02_cone_of_uncertainty.png")
plt.show()

# Print cone statistics at key points
print("\nCone of Uncertainty Statistics:")
print(f"Day 0 (Today): Price = ${S0:.2f}")
print(f"Day 63 (Quarter): Mean = ${mean_path[63]:.2f}, 90% CI = [${p_05[63]:.2f}, ${p_95[63]:.2f}]")
print(f"Day 126 (Half-Year): Mean = ${mean_path[126]:.2f}, 90% CI = [${p_05[126]:.2f}, ${p_95[126]:.2f}]")
print(f"Day 252 (1 Year): Mean = ${mean_path[252-1]:.2f}, 90% CI = [${p_05[252-1]:.2f}, ${p_95[252-1]:.2f}]")

# ============================================================================
# STEP 4: Law of Large Numbers (LLN)
# ============================================================================
print("\n[STEP 4] Demonstrating Law of Large Numbers...")

# Theoretical mean: E[S(T)] = S₀ * exp(μ*T)
theoretical_mean = S0 * np.exp(mu * T)

# Test convergence with increasing N
sim_counts = np.arange(100, 5100, 100)
estimated_means = []
estimated_stds = []

for n in sim_counts:
    # Generate n paths to T
    Z_n = np.random.standard_normal(n)
    ST_n = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_n)
    estimated_means.append(ST_n.mean())
    estimated_stds.append(ST_n.std())

# Plot LLN convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Convergence of mean
ax1 = axes[0]
ax1.plot(sim_counts, estimated_means, color='steelblue', linewidth=2.5, marker='o', markersize=4, label='Monte Carlo Estimate')
ax1.axhline(theoretical_mean, color='red', linestyle='--', linewidth=2.5, label=f'Theoretical Mean (${theoretical_mean:.2f})')
ax1.fill_between(sim_counts, theoretical_mean - 5, theoretical_mean + 5, color='red', alpha=0.1)
ax1.set_title('Law of Large Numbers: Convergence of Mean', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Simulations (N)', fontsize=11)
ax1.set_ylabel('Estimated Final Mean Price ($)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right plot: Standard deviation of estimates
ax2 = axes[1]
ax2.plot(sim_counts, estimated_stds, color='darkgreen', linewidth=2.5, marker='s', markersize=4, label='Estimated Std Dev')
ax2.set_title('Dispersion of Final Price Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Simulations (N)', fontsize=11)
ax2.set_ylabel('Standard Deviation of S(T) ($)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/05/03_law_of_large_numbers.png', dpi=300)
print("✓ Saved: 03_law_of_large_numbers.png")
plt.show()

# Analysis of LLN
error_at_100 = abs(estimated_means[0] - theoretical_mean)
error_at_5000 = abs(estimated_means[-1] - theoretical_mean)

print(f"\nLaw of Large Numbers Analysis:")
print(f"Theoretical Mean Price at T=1 year: ${theoretical_mean:.2f}")
print(f"Error with N=100 simulations: ${error_at_100:.2f}")
print(f"Error with N=5000 simulations: ${error_at_5000:.2f}")
print(f"Error reduction: {(1 - error_at_5000/error_at_100)*100:.1f}%")

# ============================================================================
# STEP 5: Key Insights
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

print(f"""
1. MULTIPLE PATHS SIMULATION:
   • We generated {N_sims} possible price paths over {T} year
   • Each path follows Geometric Brownian Motion with estimated parameters
   • Current stock: {ticker} @ ${S0:.2f}
   
2. THE CONE OF UNCERTAINTY:
   • As time increases, possible price outcomes spread out (variance increases)
   • At T=1 year: 90% of paths are between ${p_05[-1]:.2f} and ${p_95[-1]:.2f}
   • This represents the uncertainty we face in predicting future prices
   • Volatility σ={sigma*100:.2f}% drives the width of the cone
   
3. LAW OF LARGE NUMBERS:
   • With only N=100 sims, estimates can be off by ~${error_at_100:.2f}
   • With N=5000 sims, estimates converge to theoretical value
   • Running derivatives pricing with too few simulations introduces risk!
   • Typically N=10,000-100,000 needed for accurate pricing
   
4. PRACTICAL APPLICATIONS:
   • Risk Management: Use cone to estimate Value at Risk (VaR)
   • Option Pricing: Monte Carlo is essential for exotic options
   • Portfolio Analysis: Simulate multi-asset scenarios
   • Decision Making: Quantify uncertainty in financial forecasts
""")

print("=" * 80)
print("✓ Monte Carlo Simulations Demo Complete!")
print("=" * 80)
