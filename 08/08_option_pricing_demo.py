"""
Option Pricing: Monte Carlo vs Black-Scholes Demo
=================================================
Based on 08_Option_Pricing_Monte_Carlo_vs_Black_Scholes.ipynb

This demo shows:
1. Monte Carlo option pricing
2. Black-Scholes exact formula pricing
3. Comparison and convergence
4. Put-Call Parity verification
5. How parameters affect option prices
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print("=" * 80)
print("OPTION PRICING: MONTE CARLO vs BLACK-SCHOLES DEMO")
print("=" * 80)

# ============================================================================
# STEP 1: Define Pricing Functions
# ============================================================================
print("\n[STEP 1] Defining pricing functions...")

def monte_carlo_pricing(S0, K, r, sigma, T, N):
    """
    Price European Call and Put using Monte Carlo simulation.
    
    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - r: Risk-free rate
    - sigma: Volatility
    - T: Time to maturity (years)
    - N: Number of simulations
    """
    # Generate random stock prices at maturity
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    call_price = np.mean(call_payoffs) * np.exp(-r * T)
    put_price = np.mean(put_payoffs) * np.exp(-r * T)
    
    return call_price, put_price, ST

def black_scholes_pricing(S0, K, r, sigma, T):
    """
    Price European Call and Put using Black-Scholes formula.
    
    The Black-Scholes formula is exact for European options.
    Formula: C = S0*N(d1) - K*exp(-rT)*N(d2)
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    
    return call_price, put_price

print("✓ Pricing functions defined")

# ============================================================================
# STEP 2: Basic Comparison
# ============================================================================
print("\n[STEP 2] Comparing Monte Carlo vs Black-Scholes...")

S0 = 100
K = 105
r = 0.05
sigma = 0.2
T = 1.0
N = 100000

# Calculate prices
mc_call, mc_put, ST = monte_carlo_pricing(S0, K, r, sigma, T, N)
bs_call, bs_put = black_scholes_pricing(S0, K, r, sigma, T)

print(f"\nParameters:")
print(f"  Stock Price (S0): ${S0}")
print(f"  Strike Price (K): ${K}")
print(f"  Risk-free Rate (r): {r:.1%}")
print(f"  Volatility (σ): {sigma:.1%}")
print(f"  Time to Maturity (T): {T} year")
print(f"  Simulations (N): {N:,}")

print(f"\nCall Option Pricing:")
print(f"  Monte Carlo:   ${mc_call:.4f}")
print(f"  Black-Scholes: ${bs_call:.4f}")
print(f"  Difference:    ${abs(mc_call - bs_call):.6f}")

print(f"\nPut Option Pricing:")
print(f"  Monte Carlo:   ${mc_put:.4f}")
print(f"  Black-Scholes: ${bs_put:.4f}")
print(f"  Difference:    ${abs(mc_put - bs_put):.6f}")

# ============================================================================
# STEP 3: Convergence Analysis
# ============================================================================
print("\n[STEP 3] Analyzing Monte Carlo convergence...")

N_values = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000])
mc_calls = []
mc_puts = []
errors_call = []
errors_put = []

for N in N_values:
    mc_c, mc_p, _ = monte_carlo_pricing(S0, K, r, sigma, T, N)
    mc_calls.append(mc_c)
    mc_puts.append(mc_p)
    errors_call.append(abs(mc_c - bs_call))
    errors_put.append(abs(mc_p - bs_put))

# Create convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Call price convergence
ax1.plot(N_values, mc_calls, linewidth=2.5, color='steelblue', label='Monte Carlo Estimate')
ax1.axhline(bs_call, color='red', linestyle='--', linewidth=2.5, label=f'Black-Scholes (${bs_call:.4f})')
ax1.fill_between(N_values, bs_call - 0.005, bs_call + 0.005, color='red', alpha=0.1, label='±0.005 Band')
ax1.set_xscale('log')
ax1.set_title('Call Option Price Convergence', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Simulations (N)', fontsize=11)
ax1.set_ylabel('Call Price ($)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# RIGHT: Error decay
ax2.plot(N_values, errors_call, linewidth=2.5, color='steelblue', label='Call Error')
ax2.plot(N_values, errors_put, linewidth=2.5, color='coral', label='Put Error')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Monte Carlo Error Decay', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Simulations (N)', fontsize=11)
ax2.set_ylabel('Absolute Error ($)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/08/01_convergence_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_convergence_analysis.png")
plt.close()

# ============================================================================
# STEP 4: Put-Call Parity Verification
# ============================================================================
print("\n[STEP 4] Verifying Put-Call Parity...")

# Use high-precision Monte Carlo
mc_call_pcp, mc_put_pcp, _ = monte_carlo_pricing(S0, K, r, sigma, T, 500000)
bs_call_pcp, bs_put_pcp = black_scholes_pricing(S0, K, r, sigma, T)

# Put-Call Parity: C - P = S0 - K*exp(-rT)
left_side_mc = mc_call_pcp - mc_put_pcp
left_side_bs = bs_call_pcp - bs_put_pcp
right_side = S0 - K * np.exp(-r * T)

print(f"\nPut-Call Parity: C - P = S0 - K*e^(-rT)")
print(f"  Black-Scholes Left Side (C - P):  ${left_side_bs:.6f}")
print(f"  Monte Carlo Left Side (C - P):    ${left_side_mc:.6f}")
print(f"  Right Side (S0 - K*e^-rT):        ${right_side:.6f}")
print(f"  Arbitrage Error (BS):             ${abs(left_side_bs - right_side):.10f}")
print(f"  Arbitrage Error (MC):             ${abs(left_side_mc - right_side):.6f}")

# ============================================================================
# STEP 5: Impact of Volatility on Option Prices
# ============================================================================
print("\n[STEP 5] Analyzing volatility impact...")

sigmas = np.linspace(0.05, 0.5, 30)
call_prices = []
put_prices = []

for sig in sigmas:
    c, p = black_scholes_pricing(S0, K, r, sig, T)
    call_prices.append(c)
    put_prices.append(p)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(sigmas * 100, call_prices, linewidth=2.5, color='green', label='Call Option Price')
ax.plot(sigmas * 100, put_prices, linewidth=2.5, color='red', label='Put Option Price')
ax.axvline(sigma * 100, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Current σ={sigma*100:.0f}%')
ax.set_title('Option Prices vs Volatility (Vega Effect)', fontsize=12, fontweight='bold')
ax.set_xlabel('Volatility (σ) %', fontsize=11)
ax.set_ylabel('Option Price ($)', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/08/02_volatility_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_volatility_impact.png")
plt.close()

# ============================================================================
# STEP 6: Impact of Time to Maturity
# ============================================================================
print("\n[STEP 6] Analyzing time decay...")

times = np.linspace(0.05, 2, 40)
call_prices_t = []
put_prices_t = []

for t in times:
    c, p = black_scholes_pricing(S0, K, r, sigma, t)
    call_prices_t.append(c)
    put_prices_t.append(p)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(times, call_prices_t, linewidth=2.5, color='darkgreen', label='Call Option Price')
ax.plot(times, put_prices_t, linewidth=2.5, color='darkred', label='Put Option Price')
ax.axvline(T, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Current T={T} year')
ax.set_title('Option Prices vs Time to Maturity (Theta Effect)', fontsize=12, fontweight='bold')
ax.set_xlabel('Time to Maturity (T) - Years', fontsize=11)
ax.set_ylabel('Option Price ($)', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/08/03_time_decay.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_time_decay.png")
plt.close()

# ============================================================================
# STEP 7: Price Sensitivity Analysis (Greeks)
# ============================================================================
print("\n[STEP 7] Analyzing Greek sensitivities...")

stock_prices = np.linspace(80, 120, 40)
call_prices_spot = []
put_prices_spot = []

for spot in stock_prices:
    c, p = black_scholes_pricing(spot, K, r, sigma, T)
    call_prices_spot.append(c)
    put_prices_spot.append(p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: Price sensitivity (Delta)
ax1.plot(stock_prices, call_prices_spot, linewidth=2.5, color='green', label='Call Price')
ax1.plot(stock_prices, put_prices_spot, linewidth=2.5, color='red', label='Put Price')
ax1.axvline(S0, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Current S0=${S0}')
ax1.axvline(K, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Strike K=${K}')
ax1.set_title('Option Price vs Stock Price (Delta)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Stock Price ($)', fontsize=11)
ax1.set_ylabel('Option Price ($)', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# RIGHT: Intrinsic vs Time Value
call_intrinsic = [max(s - K, 0) for s in stock_prices]
call_time_value = [cp - ci for cp, ci in zip(call_prices_spot, call_intrinsic)]

ax2.fill_between(stock_prices, 0, call_intrinsic, color='darkgreen', alpha=0.5, label='Intrinsic Value')
ax2.fill_between(stock_prices, call_intrinsic, call_prices_spot, color='lightgreen', alpha=0.7, label='Time Value')
ax2.plot(stock_prices, call_prices_spot, linewidth=2.5, color='darkgreen')
ax2.axvline(S0, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_title('Call Option: Intrinsic vs Time Value', fontsize=12, fontweight='bold')
ax2.set_xlabel('Stock Price ($)', fontsize=11)
ax2.set_ylabel('Option Value ($)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sophia/Desktop/Study📄/202606 - Pioneer/Coding tests/08/04_greeks_sensitivity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_greeks_sensitivity.png")
plt.close()

# ============================================================================
# STEP 8: Summary and Insights
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: OPTION PRICING")
print("=" * 80)

print(f"""
1. MONTE CARLO VS BLACK-SCHOLES:
   • Black-Scholes = Exact formula (closed-form solution)
   • Monte Carlo = Numerical approximation via simulation
   • As N → ∞, Monte Carlo converges to Black-Scholes
   • Error scales as O(1/√N) - need 4x simulations for 2x accuracy
   
2. PUT-CALL PARITY (Arbitrage Relationship):
   • Formula: C - P = S₀ - K*e^(-rT)
   • If violated, risk-free arbitrage opportunity exists
   • Must hold exactly (within rounding) for European options
   • Prevents mispricing between call and put markets
   
3. VOLATILITY (VEGA):
   • Higher volatility → Higher option prices (both calls and puts)
   • Why? Higher volatility = more upside/downside potential
   • Options are worth more when uncertainty is higher
   • Volatility is the most important pricing input
   
4. TIME DECAY (THETA):
   • As T decreases → Call prices decrease
   • As T decreases → Put prices decrease (for OTM puts)
   • Time value erodes as expiration approaches
   • Out-of-the-money options lose value faster
   
5. DELTA (Stock Price Sensitivity):
   • Call Delta: Increases from 0 to 1 as S₀ rises
   • Put Delta: Decreases from 0 to -1 as S₀ rises
   • ATM options have Delta ≈ ±0.5
   • Delta tells you option price sensitivity per $1 stock move
   
6. WHEN TO USE WHICH METHOD:
   • Black-Scholes: Fast, exact, European options only
   • Monte Carlo: Flexible, can price exotic options, American options
   • Recommendation: Use BS for simple options, MC for complex ones
""")

print("=" * 80)
print("✓ Option Pricing Demo Complete!")
print("=" * 80)
