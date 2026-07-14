"""
Topic 09: Risk Management - Value at Risk (VaR) and Expected Shortfall (CVaR)
Demonstrates key concepts with visualizations and real-world implications
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TOPIC 09: RISK MANAGEMENT - VALUE AT RISK (VaR)")
print("="*70)

# Set random seed for reproducibility
np.random.seed(42)

# Portfolio Parameters
S0 = 100000  # $100k initial portfolio
mu = 0.08    # 8% annual return
sigma = 0.2  # 20% annual volatility
T = 1.0      # 1 year horizon
N = 50000    # number of simulations

# Generate portfolio values using Geometric Brownian Motion
Z = np.random.standard_normal(N)
ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# Calculate VaR at different confidence levels
var_90 = S0 - np.percentile(ST, 10)
var_95 = S0 - np.percentile(ST, 5)
var_99 = S0 - np.percentile(ST, 1)

print(f"\n✓ Generated {N:,} portfolio simulations over 1 year")
print(f"✓ Initial Portfolio: ${S0:,.0f}")
print(f"✓ Expected Return: {mu:.1%}, Volatility: {sigma:.1%}")
print(f"\nValue at Risk (VaR) - Maximum Expected Loss:")
print(f"  90% confidence: ${var_90:,.2f}")
print(f"  95% confidence: ${var_95:,.2f}")
print(f"  99% confidence: ${var_99:,.2f}")

# ============================================================================
# GRAPH 1: VaR Distribution and Percentile Bands
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

counts, bins, patches = ax.hist(ST, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

# Color the tail (worst 5%)
percentile_5 = np.percentile(ST, 5)
for i, patch in enumerate(patches):
    if bins[i] < percentile_5:
        patch.set_facecolor('salmon')

# Add VaR lines
ax.axvline(percentile_5, color='red', linestyle='--', linewidth=2.5, label=f'VaR (95%): ${var_95:,.0f} loss')
ax.axvline(np.percentile(ST, 10), color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'VaR (90%): ${var_90:,.0f} loss')
ax.axvline(np.percentile(ST, 1), color='darkred', linestyle='--', linewidth=2.5, label=f'VaR (99%): ${var_99:,.0f} loss')
ax.axvline(S0, color='green', linestyle='-', linewidth=2.5, label=f'Initial: ${S0:,.0f}')

ax.set_xlabel('Portfolio Value ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Portfolio Value Distribution (1-Year Horizon)\nShaded Red = Worst 5% Outcomes (VaR 95%)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('01_var_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_var_distribution.png")
plt.close()

# ============================================================================
# GRAPH 2: CVaR (Expected Shortfall) - Average of Tail Events
# ============================================================================
tail_threshold = np.percentile(ST, 5)
tail_events = ST[ST < tail_threshold]
cvar_95 = S0 - np.mean(tail_events)

fig, ax = plt.subplots(figsize=(12, 6))

counts, bins, patches = ax.hist(ST, bins=100, color='lightblue', edgecolor='black', alpha=0.7)

# Color tail events
for i, patch in enumerate(patches):
    if bins[i] < tail_threshold:
        patch.set_facecolor('crimson')

ax.axvline(tail_threshold, color='red', linestyle='--', linewidth=2.5, 
           label=f'VaR (95%): ${var_95:,.0f} loss - Minimum loss in worst 5%')
ax.axvline(np.mean(tail_events), color='darkred', linestyle='-', linewidth=2.5, 
           label=f'CVaR (95%): ${cvar_95:,.0f} loss - AVERAGE loss when VaR is breached')
ax.axvline(S0, color='green', linestyle='-', linewidth=2, alpha=0.5, label=f'Initial: ${S0:,.0f}')

ax.set_xlabel('Portfolio Value ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('CVaR (Expected Shortfall): Average Loss in Tail Events\nDark Red = When Worst-Case Actually Happens', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_cvar_expected_shortfall.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_cvar_expected_shortfall.png")
plt.close()

print(f"\nCVaR Analysis:")
print(f"  VaR (95%):  ${var_95:,.2f} ← Minimum loss threshold")
print(f"  CVaR (95%): ${cvar_95:,.2f} ← Average loss when threshold breached")
print(f"  Gap:        ${cvar_95 - var_95:,.2f} ← How much worse it actually gets")

# ============================================================================
# GRAPH 3: VaR vs Volatility (Sensitivity Analysis)
# ============================================================================
volatilities = np.linspace(0.05, 0.5, 30)
var_values = []

for vol in volatilities:
    Z_temp = np.random.standard_normal(N)
    ST_temp = S0 * np.exp((mu - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z_temp)
    var_temp = S0 - np.percentile(ST_temp, 5)
    var_values.append(var_temp)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(volatilities * 100, var_values, color='#d62728', linewidth=3, label='VaR (95%)')
ax.fill_between(volatilities * 100, 0, var_values, alpha=0.3, color='#d62728')
ax.axhline(var_95, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Current VaR (σ=20%): ${var_95:,.0f}')

ax.set_xlabel('Volatility (σ) - %', fontsize=12)
ax.set_ylabel('Value at Risk ($)', fontsize=12)
ax.set_title('Sensitivity: How Volatility Affects Risk (VaR)\nHigher volatility = Higher potential losses', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_var_volatility_sensitivity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_var_volatility_sensitivity.png")
plt.close()

# ============================================================================
# GRAPH 4: Confidence Level Trade-off
# ============================================================================
confidence_levels = np.arange(50, 100, 1)
var_at_levels = []

for conf in confidence_levels:
    percentile = 100 - conf
    var_at_level = S0 - np.percentile(ST, percentile)
    var_at_levels.append(var_at_level)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(confidence_levels, var_at_levels, color='#1f77b4', linewidth=3)
ax.scatter([90, 95, 99], [var_90, var_95, var_99], s=200, color='red', zorder=5, 
           label='Standard Confidence Levels')

# Annotations
ax.annotate(f'90%: ${var_90:,.0f}', xy=(90, var_90), xytext=(85, var_90 + 2000),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
ax.annotate(f'95%: ${var_95:,.0f}', xy=(95, var_95), xytext=(90, var_95 + 2000),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
ax.annotate(f'99%: ${var_99:,.0f}', xy=(99, var_99), xytext=(94, var_99 + 2000),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

ax.set_xlabel('Confidence Level (%)', fontsize=12)
ax.set_ylabel('Value at Risk ($)', fontsize=12)
ax.set_title('Confidence Level Trade-off: More Confidence = Larger VaR Estimate\n(Curve shows risk at different protection levels)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(50, 100)

plt.tight_layout()
plt.savefig('04_var_confidence_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_var_confidence_tradeoff.png")
plt.close()

# ============================================================================
# GRAPH 5: Parametric vs Historical VaR
# ============================================================================
# Parametric approach (what we've been doing)
returns_from_sim = (ST - S0) / S0
returns_sorted = np.sort(returns_from_sim)
parametric_var = S0 * abs(np.percentile(returns_from_sim, 5))

# Historical approach (simplified - using simulated returns as proxy)
historical_returns = returns_from_sim
historical_var = S0 * abs(np.percentile(historical_returns, 5))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Parametric VaR
ax1.hist(returns_from_sim * 100, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.percentile(returns_from_sim, 5) * 100, color='red', linestyle='--', linewidth=2.5, 
            label=f'VaR (95%): {np.percentile(returns_from_sim, 5) * 100:.2f}%')
ax1.set_xlabel('Return (%)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Parametric VaR\n(Assumes Normal Distribution)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Historical VaR
ax2.hist(historical_returns * 100, bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.axvline(np.percentile(historical_returns, 5) * 100, color='darkred', linestyle='--', linewidth=2.5,
            label=f'VaR (95%): {np.percentile(historical_returns, 5) * 100:.2f}%')
ax2.set_xlabel('Return (%)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Historical VaR\n(Uses Actual Data Distribution)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('05_parametric_vs_historical_var.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_parametric_vs_historical_var.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print(f"\n1. VaR is a PERCENTILE, not an average:")
print(f"   • 95% VaR = ${var_95:,.0f} means: 5% chance loss > this amount")
print(f"\n2. CVaR is the AVERAGE loss when VaR is breached:")
print(f"   • CVaR = ${cvar_95:,.0f} (always ≥ VaR)")
print(f"   • Gap shows risk UNDERESTIMATION by VaR alone")
print(f"\n3. Volatility directly increases VaR:")
print(f"   • At σ=5%: VaR ≈ ${S0 - np.percentile(S0 * np.exp((mu - 0.5*0.05**2)*T + 0.05*np.sqrt(T)*Z),5):,.0f}")
print(f"   • At σ=50%: VaR ≈ ${S0 - np.percentile(S0 * np.exp((mu - 0.5*0.5**2)*T + 0.5*np.sqrt(T)*Z),5):,.0f}")
print(f"\n4. Confidence level choice is a judgment call:")
print(f"   • Regulators often require 99% VaR")
print(f"   • Risk managers often use 95% VaR for decisions")
print(f"\n5. 2008 Financial Crisis:")
print(f"   • Models assumed normal distribution (parametric)")
print(f"   • Reality: FAT TAILS (extreme events more common)")
print(f"   • Lesson: Use CVaR + historical data, not just VaR")
print("\n" + "="*70)
