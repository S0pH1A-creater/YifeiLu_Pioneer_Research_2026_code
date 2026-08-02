"""
Introduction to Options Demo
=====================================
Based on 07_Introduction_to_Options.ipynb

This demo shows:
1. Call and Put option payoffs at expiration
2. Moneyness (In-The-Money, At-The-Money, Out-of-The-Money)
3. Protective Put strategy (risk transformation)
4. Real financial scenarios with various stock prices
"""


from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

print("=" * 80)
print("INTRODUCTION TO OPTIONS DEMO")
print("=" * 80)

# ============================================================================
# STEP 1: Setup - Get Real Stock Data
# ============================================================================
print("\n[STEP 1] Downloading real stock data to estimate volatility...")

ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

data = None
max_retries = 2
for attempt in range(max_retries):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            print(f"✓ Downloaded {len(data)} trading days for {ticker}")
            break
    except Exception as e:
        print(f"✗ API attempt {attempt+1} failed")
        import time
        time.sleep(2)

# Fallback to synthetic data
if data is None or len(data) == 0:
    print("Using synthetic market data...")
    import pandas as pd
    n_days = 252
    drift = 0.0005
    volatility = 0.015
    price_init = 150
    returns = np.random.normal(drift, volatility, n_days)
    prices = price_init * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=end_date, periods=n_days, freq='B')
    data = pd.DataFrame({'Adj Close': prices}, index=dates)

current_price = data['Adj Close'].iloc[-1]
print(f"Current {ticker} Price: ${current_price:.2f}")

# ============================================================================
# STEP 2: Basic Call and Put Option Payoffs
# ============================================================================
print("\n[STEP 2] Visualizing Call and Put option payoffs...")

# Setup parameters
ST = np.linspace(50, 200, 300)  # Range of stock prices at expiration
K = 120  # Strike price
call_premium = 5
put_premium = 6

# Calculate payoffs
call_payoff = np.maximum(ST - K, 0)  # Intrinsic value
put_payoff = np.maximum(K - ST, 0)

call_profit = call_payoff - call_premium
put_profit = put_payoff - put_premium

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# ===== TOP LEFT: Call Option Payoff and Profit =====
ax1.plot(ST, call_payoff, linewidth=2.5, color='darkgreen', label='Payoff (intrinsic value)')
ax1.plot(ST, call_profit, linewidth=2.5, color='lime', label=f'Profit (payoff - ${call_premium} premium)')
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.axvline(K, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Strike K=${K}')
ax1.axvline(K + call_premium, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Break-even=${K+call_premium:.0f}')
ax1.fill_between(ST, 0, call_profit, where=(call_profit > 0), color='green', alpha=0.2, label='Profit Zone')
ax1.fill_between(ST, 0, call_profit, where=(call_profit <= 0), color='red', alpha=0.2, label='Loss Zone')
ax1.set_title('Call Option: Right to BUY at Strike K', fontsize=12, fontweight='bold')
ax1.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax1.set_ylabel('Profit / Loss ($)', fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(50, 200)

# ===== TOP RIGHT: Put Option Payoff and Profit =====
ax2.plot(ST, put_payoff, linewidth=2.5, color='darkred', label='Payoff (intrinsic value)')
ax2.plot(ST, put_profit, linewidth=2.5, color='salmon', label=f'Profit (payoff - ${put_premium} premium)')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.axvline(K, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Strike K=${K}')
ax2.axvline(K - put_premium, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Break-even=${K-put_premium:.0f}')
ax2.fill_between(ST, 0, put_profit, where=(put_profit > 0), color='green', alpha=0.2, label='Profit Zone')
ax2.fill_between(ST, 0, put_profit, where=(put_profit <= 0), color='red', alpha=0.2, label='Loss Zone')
ax2.set_title('Put Option: Right to SELL at Strike K', fontsize=12, fontweight='bold')
ax2.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax2.set_ylabel('Profit / Loss ($)', fontsize=11)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(50, 200)

# ===== BOTTOM LEFT: Option Moneyness =====
ax3.plot(ST, call_profit, linewidth=2.5, color='black', label='Call Profit')
ax3.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax3.axvline(K, color='purple', linestyle='--', linewidth=2, alpha=0.8, label=f'ATM Strike K=${K}')

# Shade regions for moneyness
ax3.fill_between(ST, -10, call_profit, where=(ST < K - 10), color='red', alpha=0.15, label='Deep OTM')
ax3.fill_between(ST, -10, call_profit, where=((ST >= K - 10) & (ST < K)), color='orange', alpha=0.15, label='OTM')
ax3.fill_between(ST, -10, call_profit, where=(ST >= K), color='green', alpha=0.15, label='ITM')

ax3.set_title('Call Option Moneyness (ITM / ATM / OTM)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax3.set_ylabel('Profit / Loss ($)', fontsize=11)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(50, 200)
ax3.set_ylim(-call_premium - 5, 50)

# ===== BOTTOM RIGHT: Call vs Put =====
ax4.plot(ST, call_profit, linewidth=2.5, color='green', label='Long Call')
ax4.plot(ST, put_profit, linewidth=2.5, color='red', label='Long Put')
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax4.axvline(K, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Strike K=${K}')
ax4.set_title('Comparison: Call vs Put Options', fontsize=12, fontweight='bold')
ax4.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax4.set_ylabel('Profit / Loss ($)', fontsize=11)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(50, 200)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '01_call_put_basics.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: 01_call_put_basics.png")
plt.close()

# ============================================================================
# STEP 3: Protective Put Strategy (Risk Transformation)
# ============================================================================
print("\n[STEP 3] Visualizing Protective Put strategy...")

S0 = 120  # Current stock price
put_strike = 110
put_premium = 6

# Calculate profits
stock_only_profit = ST - S0
put_payoff_pp = np.maximum(put_strike - ST, 0)
protective_put_profit = (ST - S0) + put_payoff_pp - put_premium

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== LEFT: Protective Put Strategy =====
ax1.plot(ST, stock_only_profit, linewidth=2.5, color='gray', linestyle='--', label='Stock Only (unprotected)', alpha=0.7)
ax1.plot(ST, protective_put_profit, linewidth=2.5, color='blue', label='Stock + Put Option (Protected)')
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.axvline(S0, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Current Price S0=${S0}')
ax1.axvline(put_strike, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Put Strike=${put_strike}')

# Shade the protected region
protected_loss = put_strike - S0 - put_premium
ax1.fill_between(ST, protected_loss, stock_only_profit, where=(ST < put_strike), color='lightblue', alpha=0.3, label='Protected Downside')
ax1.fill_between(ST, stock_only_profit, protective_put_profit, color='green', alpha=0.1, label='Unchanged Upside')

ax1.set_title('Protective Put: Risk Transformation', fontsize=12, fontweight='bold')
ax1.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax1.set_ylabel('Profit / Loss ($)', fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(80, 160)

# Add annotations
max_loss = put_strike - S0 - put_premium
ax1.annotate(f'Max Loss: ${max_loss:.2f}', xy=(put_strike - 5, max_loss), xytext=(put_strike - 25, max_loss - 10),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, color='red', fontweight='bold')

# ===== RIGHT: Comparison of Strategies =====
# Long Call strategy (similar payoff to protective put)
call_strike = 110
call_premium = 5
long_call_profit = np.maximum(ST - call_strike, 0) - call_premium

ax2.plot(ST, protective_put_profit, linewidth=2.5, color='blue', label='Protective Put: Stock + Put')
ax2.plot(ST, long_call_profit, linewidth=2.5, color='purple', linestyle='--', label='Long Call (for comparison)')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.axvline(S0, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

ax2.set_title('Strategy Comparison: Protective Put vs Call', fontsize=12, fontweight='bold')
ax2.set_xlabel('Stock Price at Expiration (ST)', fontsize=11)
ax2.set_ylabel('Profit / Loss ($)', fontsize=11)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(80, 160)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '02_protective_put.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: 02_protective_put.png")
plt.close()

# ============================================================================
# STEP 4: Real-World Scenario Analysis
# ============================================================================
print("\n[STEP 4] Real-world scenario: Options on actual stock price...")

# Use current stock price
current_stock = current_price
atm_strike = round(current_stock / 5) * 5  # Round to nearest 5
otm_strike = atm_strike + 10
itm_strike = atm_strike - 10

price_range = np.linspace(current_stock * 0.7, current_stock * 1.4, 300)

# Different strikes
atm_call = np.maximum(price_range - atm_strike, 0) - 3
itm_call = np.maximum(price_range - itm_strike, 0) - 5
otm_call = np.maximum(price_range - otm_strike, 0) - 2

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(price_range, itm_call, linewidth=2.5, color='darkgreen', label=f'ITM Call (K=${itm_strike:.0f})')
ax.plot(price_range, atm_call, linewidth=2.5, color='blue', label=f'ATM Call (K=${atm_strike:.0f})')
ax.plot(price_range, otm_call, linewidth=2.5, color='red', label=f'OTM Call (K=${otm_strike:.0f})')

ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.axvline(current_stock, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Current Price: ${current_stock:.2f}')

ax.fill_between(price_range, 0, atm_call, where=(atm_call > 0), color='green', alpha=0.1)
ax.fill_between(price_range, 0, atm_call, where=(atm_call <= 0), color='red', alpha=0.1)

ax.set_title(f'{ticker} Call Options: ITM vs ATM vs OTM\nCurrent Price: ${current_stock:.2f}', fontsize=12, fontweight='bold')
ax.set_xlabel('Stock Price at Expiration', fontsize=11)
ax.set_ylabel('Call Profit / Loss ($)', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '03_real_world_moneyness.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_real_world_moneyness.png")
plt.close()

# ============================================================================
# STEP 5: Summary and Insights
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: INTRODUCTION TO OPTIONS")
print("=" * 80)

print(f"""
1. CALL OPTIONS:
   • Right to BUY at strike price K
   • Profit when ST > K + premium (break-even)
   • Maximum loss: Premium paid
   • Unlimited profit potential
   
2. PUT OPTIONS:
   • Right to SELL at strike price K
   • Profit when ST < K - premium (break-even)
   • Maximum loss: Premium paid
   • Maximum profit: Strike - Premium
   
3. MONEYNESS:
   • ITM (In-The-Money): Has intrinsic value
   • ATM (At-The-Money): Strike ≈ Current price
   • OTM (Out-of-The-Money): Zero intrinsic value
   
4. PROTECTIVE PUT (Insurance Strategy):
   • Buy stock + Buy put option
   • Caps maximum loss at (K - S0 - premium)
   • Preserves unlimited upside
   • Real-world use: Portfolio protection during crises
   
5. REAL EXAMPLE ({ticker}):
   • Current Price: ${current_stock:.2f}
   • ATM Strike: ${atm_strike:.0f}
   • ITM Strike: ${itm_strike:.0f}
   • OTM Strike: ${otm_strike:.0f}

6. WHEN TO USE EACH STRATEGY:
   • Long Call: Expect stock UP, limited risk
   • Long Put: Expect stock DOWN, limited risk
   • Protective Put: Own stock, want downside protection
   • Call Spread: Limited risk AND limited profit
""")

print("=" * 80)
print("✓ Options Demo Complete!")
print("=" * 80)
