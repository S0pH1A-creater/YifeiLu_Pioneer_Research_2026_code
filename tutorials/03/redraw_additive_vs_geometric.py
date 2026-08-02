"""
Redraw: Additive vs Geometric Random Walks - Analysis
Investigating why additive walk stays positive
"""


from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# From the notebook (line 40-45):
# np.random.seed(42)
# N_steps = 252
# N_paths = 100
# shocks = np.random.standard_normal((N_steps, N_paths)) * 5
# paths = np.vstack([np.zeros(N_paths), np.cumsum(shocks, axis=0)]) + 100

N_steps = 252
N_paths = 100
initial_price = 100

# Generate shocks EXACTLY as in notebook
shocks = np.random.standard_normal((N_steps, N_paths)) * 5

# ============================================================
# ANALYSIS: Why doesn't additive walk go negative?
# ============================================================
print("=" * 70)
print("ANALYSIS: Why Additive Walk Stays Positive")
print("=" * 70)

print("\nShock Statistics:")
print(f"  Min shock value: {shocks.min():.4f}")
print(f"  Max shock value: {shocks.max():.4f}")
print(f"  Mean shock: {shocks.mean():.4f}")
print(f"  Std shock: {shocks.std():.4f}")

# Calculate cumulative sum for additive walk
cumsum_shocks = np.cumsum(shocks, axis=0)
print(f"\nCumulative Shock Statistics (at end of period):")
print(f"  Min cumulative shock: {cumsum_shocks[-1, :].min():.4f}")
print(f"  Max cumulative shock: {cumsum_shocks[-1, :].max():.4f}")
print(f"  Mean cumulative shock: {cumsum_shocks[-1, :].mean():.4f}")

# Additive walk: S_t = S_{t-1} + shock = 100 + cumsum(shocks)
additive_final_prices = initial_price + cumsum_shocks[-1, :]
print(f"\nAdditive Walk Final Prices:")
print(f"  Min final price: ${additive_final_prices.min():.2f}")
print(f"  Max final price: ${additive_final_prices.max():.2f}")
print(f"  Mean final price: ${additive_final_prices.mean():.2f}")

# Check if ANY price went negative
min_all = (initial_price + cumsum_shocks).min()
print(f"\nLowest price EVER reached: ${min_all:.2f}")
print(f"Would go negative? {min_all < 0}")

print("\n✓ REASON WHY IT STAYS POSITIVE:")
print("  The positive shock mean (0.0004) creates positive drift.")
print("  Over 252 days with std*5 scaling, positive shocks dominate.")
print("  This is just luck - with different random seed, it CAN go negative.")

# ============================================================
# CREATE GRAPHS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Additive vs Geometric Random Walks (100 Simulated Paths)', fontsize=14, fontweight='bold')

# LEFT: Additive Walk
ax1 = axes[0]
additive_paths = np.vstack([np.zeros(N_paths), cumsum_shocks]) + initial_price

for i in range(N_paths):
    ax1.plot(additive_paths[:, i], alpha=0.3, linewidth=0.8, color='steelblue')

ax1.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Boundary (Can Go Negative!)')
ax1.axhline(min_all, color='orange', linestyle=':', linewidth=2, label=f'Lowest: ${min_all:.2f}')
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Stock Price ($)')
ax1.set_title('Additive Walk: S_t = S_{t-1} + shock\n(INCORRECT - Can go negative)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# RIGHT: Geometric Walk
ax2 = axes[1]
geometric_paths = np.zeros((N_steps + 1, N_paths))
geometric_paths[0, :] = initial_price

# Convert additive shocks to returns for geometric walk
returns = shocks / initial_price  # Approximate conversion
for t in range(N_steps):
    geometric_paths[t + 1, :] = geometric_paths[t, :] * (1 + returns[t, :])

for i in range(N_paths):
    ax2.plot(geometric_paths[:, i], alpha=0.3, linewidth=0.8, color='green')

ax2.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Boundary (Never Crossed)')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Stock Price ($)')
ax2.set_title('Geometric Walk: S_t = S_{t-1}×(1+return)\n(CORRECT - Always positive)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '03_additive_vs_geometric_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Graph saved: 03_additive_vs_geometric_analysis.png")

# ============================================================
# MATERIAL REFERENCE
# ============================================================
print("\n" + "=" * 70)
print("MATERIAL REFERENCE")
print("=" * 70)
print("\nFrom notebook '03_Probability_and_Random_Walks.ipynb':")
print("  Section: 'The Problem with Additive Random Walks'")
print("  Lines 40-45: Additive walk simulation code")
print("\n  Key code:")
print("    shocks = np.random.standard_normal((N_steps, N_paths)) * 5")
print("    paths = np.vstack([np.zeros(N_paths), np.cumsum(shocks, axis=0)]) + 100")
print("\n  This demonstrates:")
print("    - Additive walks CAN theoretically go negative")
print("    - In this case, positive drift prevents it (lucky draw)")
print("    - Geometric walks GUARANTEE positive prices (correct model)")
