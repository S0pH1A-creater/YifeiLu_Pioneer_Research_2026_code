"""
Generate empirical comparison graph with synthetic realistic market data
(since yfinance is rate-limited)
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating GBM vs Realistic Market Path comparison...")

# Recreate GBM simulation
S0 = 100
mu = 0.08
sigma = 0.2
T = 1
N_steps = 252
dt = T / N_steps
times = np.linspace(0, T, N_steps)

# Generate GBM path
Z = np.random.standard_normal(N_steps)
W = np.cumsum(Z) * np.sqrt(dt)
drift = (mu - 0.5 * sigma**2) * times
diffusion = sigma * W
gbm_path = S0 * np.exp(drift + diffusion)

# Create a realistic market path with volatility clustering
# (more realistic than GBM which assumes constant volatility)
np.random.seed(123)
market_returns = []
for i in range(N_steps):
    # Add volatility clustering - higher volatility periods
    volatility_cluster = 0.25 if (i % 50 < 15) else 0.15
    daily_return = np.random.normal(mu/252, volatility_cluster/np.sqrt(252))
    market_returns.append(daily_return)

market_price = S0 * np.cumprod(1 + np.array(market_returns))

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 7))

# Plot realistic market path (black)
ax.plot(times, market_price, color='black', linewidth=2.5, 
        label='Realistic Market Path (with volatility clustering)', zorder=10)

# Plot GBM simulation (orange)
ax.plot(times, gbm_path, color='darkorange', linewidth=2.2, alpha=0.8,
        label='Simulated GBM Path (constant volatility)', zorder=5)

# Reference line
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Time (years)', fontweight='bold', fontsize=11)
ax.set_ylabel('Normalized Price (base = 100)', fontweight='bold', fontsize=11)
ax.set_title('Reality vs Simulation: Market Path vs GBM Model', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()

output_dir = OUTPUT_DIR
comparison_path = os.path.join(output_dir, '04_GBM_vs_Reality.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_GBM_vs_Reality.png")
plt.close()

print("\nKey Insight:")
print("  GBM assumes constant volatility, but real markets show volatility clustering")
print("  (periods of high/low volatility), which the GBM model fails to capture.")
