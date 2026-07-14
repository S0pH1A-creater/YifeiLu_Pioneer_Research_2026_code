"""
Redraw Monte Carlo Distribution Graph - Without Bar Margins
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
N_steps = 252  # One trading year
N_paths_large = 10000
initial_price = 100
daily_return_mean = 0.000803
daily_return_std = 0.011796

# Generate random shocks
shocks_large = np.random.normal(daily_return_mean, daily_return_std, (N_steps, N_paths_large))

# Create geometric paths
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
    
    # Create histogram WITHOUT margins on bars (edgecolor='none')
    ax.hist(final_prices, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='none')
    
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

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/03_monte_carlo_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Monte Carlo analysis graph redrawn and saved without bar margins")
print("  File: 03_monte_carlo_analysis.png")
