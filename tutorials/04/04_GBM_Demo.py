"""
Demo: Geometric Brownian Motion (GBM) - Concepts Learning
================================================================
This script demonstrates the key concepts from 04_Geometric_Brownian_Motion.ipynb:

1. Decomposing GBM into drift and diffusion components
2. Understanding the mathematical structure
3. Comparing simulated GBM paths with real market data

Reference: The GBM formula decomposition shows how deterministic drift 
and stochastic diffusion combine into the exponential path.
"""


from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

print("="*70)
print("GEOMETRIC BROWNIAN MOTION (GBM) DEMO")
print("="*70)

# ============================================================================
# PART 1: GBM PARAMETER SETUP
# ============================================================================
print("\n[PART 1] Setting Up GBM Parameters")
print("-" * 70)

S0 = 100          # Initial stock price
mu = 0.08         # Expected annual return (drift)
sigma = 0.2       # Volatility (annualized)
T = 1             # Time horizon (1 year)
N_steps = 252     # Trading days in a year
dt = T / N_steps  # Time step size

print(f"Initial Price (S0):        ${S0:.2f}")
print(f"Drift (μ):                 {mu:.1%}")
print(f"Volatility (σ):            {sigma:.1%}")
print(f"Time Horizon (T):          {T} year")
print(f"Number of Steps (N):       {N_steps} trading days")
print(f"Time Step (dt):            {dt:.6f}")

# ============================================================================
# PART 2: DECOMPOSE GBM INTO DRIFT AND DIFFUSION
# ============================================================================
print("\n[PART 2] Decomposing GBM into Drift and Diffusion Components")
print("-" * 70)
print("GBM Formula: S_T = S_0 × exp((μ - σ²/2)T + σ√T Z)")
print("  • Drift Component:     (μ - σ²/2)T      [deterministic]")
print("  • Diffusion Component: σ√T Z            [stochastic/random]")

# Generate random normal variables (Brownian increments)
Z = np.random.standard_normal(N_steps)
times = np.linspace(0, T, N_steps)

# Cumulative Brownian Motion path
W = np.cumsum(Z) * np.sqrt(dt)

# 1. Pure Drift Component (Deterministic) - no randomness
drift_component = (mu - 0.5 * sigma**2) * times

# 2. Pure Diffusion Component (Stochastic) - with randomness
diffusion_component = sigma * W

# 3. Combined Exponent
exponent = drift_component + diffusion_component

# 4. Final GBM Path
gbm_path = S0 * np.exp(exponent)

print(f"\nDrift component range:     [{drift_component.min():.4f}, {drift_component.max():.4f}]")
print(f"Diffusion component range: [{diffusion_component.min():.4f}, {diffusion_component.max():.4f}]")
print(f"GBM path range:            [${gbm_path.min():.2f}, ${gbm_path.max():.2f}]")

# ============================================================================
# PART 3: VISUALIZE DRIFT, DIFFUSION, AND COMBINED PATH
# ============================================================================
print("\n[PART 3] Creating Decomposition Visualization")
print("-" * 70)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('GBM Decomposition: Drift + Diffusion = Path', fontsize=14, fontweight='bold')

# Plot 1: Deterministic Drift
axes[0].plot(times, drift_component, color='#2E7D32', linewidth=2.5)
axes[0].fill_between(times, drift_component, alpha=0.2, color='#2E7D32')
axes[0].set_ylabel('Drift Value', fontweight='bold')
axes[0].set_title('1. Deterministic Drift: (μ - σ²/2)T', fontsize=11, loc='left')
axes[0].grid(alpha=0.3, linestyle='--')

# Plot 2: Stochastic Diffusion
axes[1].plot(times, diffusion_component, color='#6A1B9A', linewidth=1.5, alpha=0.8)
axes[1].fill_between(times, diffusion_component, alpha=0.15, color='#6A1B9A')
axes[1].set_ylabel('Diffusion Value', fontweight='bold')
axes[1].set_title('2. Stochastic Diffusion: σ√T Z (random noise)', fontsize=11, loc='left')
axes[1].grid(alpha=0.3, linestyle='--')

# Plot 3: Combined Exponential Path
axes[2].plot(times, gbm_path, color='#FF6F00', linewidth=2, label='GBM Path')
axes[2].axhline(y=S0, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'Initial Price (${S0})')
axes[2].fill_between(times, gbm_path, S0, alpha=0.1, color='#FF6F00')
axes[2].set_ylabel('Stock Price ($)', fontweight='bold')
axes[2].set_xlabel('Time (years)', fontweight='bold')
axes[2].set_title('3. Combined Exponential Path: S_T = S_0 × exp(drift + diffusion)', fontsize=11, loc='left')
axes[2].legend(loc='best', fontsize=9)
axes[2].grid(alpha=0.3, linestyle='--')

plt.tight_layout()

# Save the figure
output_dir = OUTPUT_DIR
decomp_path = os.path.join(output_dir, '04_GBM_Decomposition.png')
plt.savefig(decomp_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_GBM_Decomposition.png")
plt.close()

# ============================================================================
# PART 4: GENERATE MULTIPLE GBM PATHS (Monte Carlo Simulation)
# ============================================================================
print("\n[PART 4] Simulating Multiple GBM Paths")
print("-" * 70)

n_simulations = 100
gbm_paths = np.zeros((n_simulations, N_steps))

print(f"Running {n_simulations} simulations...")
for i in range(n_simulations):
    Z_sim = np.random.standard_normal(N_steps)
    W_sim = np.cumsum(Z_sim) * np.sqrt(dt)
    exponent_sim = (mu - 0.5 * sigma**2) * times + sigma * W_sim
    gbm_paths[i, :] = S0 * np.exp(exponent_sim)

# Calculate statistics
mean_path = np.mean(gbm_paths, axis=0)
std_path = np.std(gbm_paths, axis=0)
percentile_5 = np.percentile(gbm_paths, 5, axis=0)
percentile_95 = np.percentile(gbm_paths, 95, axis=0)

print(f"Final price statistics across {n_simulations} simulations:")
print(f"  Mean:  ${mean_path[-1]:.2f}")
print(f"  Std:   ${std_path[-1]:.2f}")
print(f"  5th percentile:  ${percentile_5[-1]:.2f}")
print(f"  95th percentile: ${percentile_95[-1]:.2f}")

# ============================================================================
# PART 5: VISUALIZE MONTE CARLO PATHS
# ============================================================================
print("\n[PART 5] Plotting Monte Carlo Simulation Results")
print("-" * 70)

fig, ax = plt.subplots(figsize=(12, 7))

# Plot individual paths with low opacity to avoid clutter
for i in range(n_simulations):
    ax.plot(times, gbm_paths[i, :], alpha=0.15, color='steelblue', linewidth=0.8)

# Plot statistics
ax.plot(times, mean_path, color='darkred', linewidth=2.5, label='Mean Path', zorder=5)
ax.fill_between(times, percentile_5, percentile_95, alpha=0.2, color='coral', 
                label='90% Confidence Band (5th-95th percentile)', zorder=3)
ax.axhline(y=S0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Initial Price (${S0})')

ax.set_xlabel('Time (years)', fontweight='bold', fontsize=11)
ax.set_ylabel('Stock Price ($)', fontweight='bold', fontsize=11)
ax.set_title(f'Monte Carlo GBM Simulation ({n_simulations} paths)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
mc_path = os.path.join(output_dir, '04_GBM_MonteCarlo.png')
plt.savefig(mc_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_GBM_MonteCarlo.png")
plt.close()

# ============================================================================
# PART 6: DOWNLOAD REAL FINANCIAL DATA AND COMPARE
# ============================================================================
print("\n[PART 6] Downloading Real Market Data for Comparison")
print("-" * 70)

try:
    # Download AAPL data for the past year
    ticker = 'AAPL'
    print(f"Downloading {ticker} data from yfinance...")
    aapl = yf.Ticker(ticker).history(period='1y')['Close']
    
    # Normalize AAPL to start at 100 (same as S0)
    aapl_normalized = (aapl / aapl.iloc[0]) * 100
    
    print(f"✓ Downloaded {len(aapl)} days of data")
    print(f"AAPL price range: ${aapl.min():.2f} - ${aapl.max():.2f}")
    print(f"AAPL normalized range: ${aapl_normalized.min():.2f} - ${aapl_normalized.max():.2f}")
    
    # ========================================================================
    # PART 7: EMPIRICAL COMPARISON - SIMULATED VS REAL
    # ========================================================================
    print("\n[PART 7] Comparing Simulated GBM with Real Market Data")
    print("-" * 70)
    print("Reference: GBM assumes constant volatility and log-normal distribution.")
    print("Real data may show volatility clustering and non-normal distributions.")
    
    # Create time axis matching the data length
    time_axis_real = np.linspace(0, 1, len(aapl_normalized))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot simulated paths (with opacity)
    for i in range(min(50, n_simulations)):  # Only plot 50 paths to avoid clutter
        ax.plot(times, gbm_paths[i, :], alpha=0.08, color='steelblue', linewidth=0.7)
    
    # Plot real data
    ax.plot(time_axis_real, aapl_normalized.values, color='black', linewidth=2.5, 
            label=f'Actual {ticker} (normalized)', zorder=10)
    
    # Plot mean of simulations
    ax.plot(times, mean_path, color='darkred', linewidth=2, linestyle='--',
            label='GBM Mean Path', zorder=5)
    
    ax.set_xlabel('Time (years)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Normalized Price (base = 100)', fontweight='bold', fontsize=11)
    ax.set_title(f'Reality vs Simulation: {ticker} vs GBM Model', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, '04_GBM_vs_Reality.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_GBM_vs_Reality.png")
    plt.close()
    
except Exception as e:
    print(f"⚠ Could not download real data: {e}")

# ============================================================================
# PART 8: ANALYSIS AND INSIGHTS
# ============================================================================
print("\n[PART 8] Key Insights and Analysis")
print("="*70)

final_prices_simulated = gbm_paths[:, -1]
mean_return = np.mean((final_prices_simulated - S0) / S0) * 100
std_return = np.std((final_prices_simulated - S0) / S0) * 100

print(f"\nExpected Return: {mu*100:.1f}%")
print(f"Simulated Mean Return: {mean_return:.2f}%")
print(f"Simulated Std Dev of Returns: {std_return:.2f}%")
print(f"\nProbability of profit (P(S_T > S_0)):")
print(f"  {np.sum(final_prices_simulated > S0) / n_simulations * 100:.1f}%")

print("\n" + "="*70)
print("DEMO COMPLETED SUCCESSFULLY")
print("="*70)
print("\nGenerated files:")
print("  • 04_GBM_Decomposition.png  - Shows drift, diffusion, and combined path")
print("  • 04_GBM_MonteCarlo.png     - Shows 100 simulated paths with statistics")
print("  • 04_GBM_vs_Reality.png     - Compares simulation with real AAPL data")
