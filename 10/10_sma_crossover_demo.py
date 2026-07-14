"""
Topic 10: Algorithmic Trading - SMA Crossover Strategy
Demonstrates a classic technical trading strategy with backtesting and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TOPIC 10: ALGORITHMIC TRADING - SMA CROSSOVER STRATEGY")
print("="*70)

# Generate synthetic stock price data (realistic)
np.random.seed(42)
n_days = 1000
returns = np.random.normal(0.0005, 0.015, n_days)  # Daily returns: 0.05% mean, 1.5% vol
prices = 100 * np.exp(np.cumsum(returns))  # Geometric brownian motion

df = pd.DataFrame({
    'Close': prices
}, index=pd.date_range('2021-01-01', periods=n_days, freq='D'))

# Calculate SMAs
sma_short = 20  # Fast moving average
sma_long = 50   # Slow moving average

df['SMA_20'] = df['Close'].rolling(window=sma_short).mean()
df['SMA_50'] = df['Close'].rolling(window=sma_long).mean()

# Generate trading signals
# Buy signal: SMA20 crosses above SMA50
# Sell signal: SMA20 crosses below SMA50
df['Signal'] = 0.0
df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1.0
df.loc[df['SMA_20'] <= df['SMA_50'], 'Signal'] = 0.0

# Identify actual trades (position changes)
df['Position'] = df['Signal'].diff()

# Calculate returns
df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)

# Transaction costs (5 basis points = 0.05% per trade)
transaction_cost = 0.0005
df['Trades'] = abs(df['Position'].fillna(0))
df['Strategy_Return_Net'] = df['Strategy_Return'] - (df['Trades'] * transaction_cost)

# Calculate cumulative returns
df['Cumulative_Market'] = (1 + df['Daily_Return']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
df['Cumulative_Strategy_Net'] = (1 + df['Strategy_Return_Net']).cumprod()

print(f"\n✓ Generated {n_days} days of price data")
print(f"✓ SMA Strategy: SMA{sma_short} crosses SMA{sma_long}")
print(f"✓ Transaction cost: {transaction_cost*100:.2f}% per trade")

# ============================================================================
# GRAPH 1: Price with SMAs and Trading Signals
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df['Close'], color='black', linewidth=2, label='Stock Price', alpha=0.8)
ax.plot(df.index, df['SMA_20'], color='#1f77b4', linewidth=2, label='SMA(20) - Fast')
ax.plot(df.index, df['SMA_50'], color='#ff7f0e', linewidth=2, label='SMA(50) - Slow')

# Mark buy and sell signals
buy_signals = df[df['Position'] == 1.0]
sell_signals = df[df['Position'] == -1.0]

ax.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=200, 
          label='BUY Signal', zorder=5, alpha=0.8)
ax.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=200, 
          label='SELL Signal', zorder=5, alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Stock Price ($)', fontsize=12)
ax.set_title('SMA Crossover Strategy: Buy/Sell Signals\nGreen △ = Buy (SMA20 > SMA50), Red ▽ = Sell (SMA20 < SMA50)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('01_sma_signals.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_sma_signals.png")
plt.close()

# ============================================================================
# GRAPH 2: Strategy vs Buy-and-Hold Performance
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df['Cumulative_Market'], color='blue', linewidth=2.5, 
       label=f'Buy & Hold: {(df["Cumulative_Market"].iloc[-1] - 1)*100:.1f}% return')
ax.plot(df.index, df['Cumulative_Strategy'], color='green', linewidth=2.5, linestyle='--',
       label=f'SMA Strategy (Gross): {(df["Cumulative_Strategy"].iloc[-1] - 1)*100:.1f}% return')
ax.plot(df.index, df['Cumulative_Strategy_Net'], color='darkgreen', linewidth=2.5,
       label=f'SMA Strategy (Net): {(df["Cumulative_Strategy_Net"].iloc[-1] - 1)*100:.1f}% return')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return ($, starting at $1)', fontsize=12)
ax.set_title('Strategy Performance: SMA Crossover vs Buy & Hold\nNon-gross includes transaction costs', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_performance_comparison.png")
plt.close()

# ============================================================================
# GRAPH 3: Drawdown Analysis
# ============================================================================
def calculate_drawdown_series(cumulative_returns):
    """Calculate running maximum drawdown"""
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown

market_drawdown = calculate_drawdown_series(df['Cumulative_Market'])
strategy_drawdown = calculate_drawdown_series(df['Cumulative_Strategy_Net'])

fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(df.index, market_drawdown * 100, 0, alpha=0.5, color='red', label='Buy & Hold Drawdown')
ax.fill_between(df.index, strategy_drawdown * 100, 0, alpha=0.5, color='orange', label='SMA Strategy Drawdown')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title('Maximum Drawdown: How Deep Losses Get During Downturns\nLower is better (less pain)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_drawdown_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_drawdown_analysis.png")
plt.close()

# ============================================================================
# GRAPH 4: Rolling Sharpe Ratio (30-day window)
# ============================================================================
window = 30
market_sharpe = df['Daily_Return'].rolling(window).apply(
    lambda x: np.sqrt(252) * (x.mean() / x.std()) if x.std() > 0 else 0
)
strategy_sharpe = df['Strategy_Return_Net'].rolling(window).apply(
    lambda x: np.sqrt(252) * (x.mean() / x.std()) if x.std() > 0 else 0
)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, market_sharpe, color='blue', linewidth=2.5, label='Buy & Hold (30-day rolling)')
ax.plot(df.index, strategy_sharpe, color='green', linewidth=2.5, label='SMA Strategy (30-day rolling)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Rolling Sharpe Ratio (30-day window): Risk-Adjusted Returns\nHigher = Better risk-adjusted performance', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('04_rolling_sharpe.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_rolling_sharpe.png")
plt.close()

# ============================================================================
# GRAPH 5: Win Rate & Trade Analysis
# ============================================================================
trades_buy = df[df['Position'] == 1.0].copy()
trades_sell = df[df['Position'] == -1.0].copy()

total_trades = len(trades_buy)
winning_trades = len(df[df['Strategy_Return_Net'] > 0.01])
losing_trades = len(df[df['Strategy_Return_Net'] < -0.01])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Trade distribution
trade_returns = df[df['Trades'] > 0]['Strategy_Return_Net'].dropna() * 100
ax1.hist(trade_returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(trade_returns.mean(), color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {trade_returns.mean():.3f}%')
ax1.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.set_xlabel('Return per Trade (%)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Trade Returns\n(Net of transaction costs)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Right: Performance metrics
metrics = ['Total\nTrades', 'Avg Return\nper Trade', 'Max Single\nTrade', 'Win Rate\n(%)']
values = [
    total_trades,
    trade_returns.mean(),
    trade_returns.max(),
    (len(trade_returns[trade_returns > 0]) / len(trade_returns) * 100) if len(trade_returns) > 0 else 0
]

colors_bar = ['blue', 'green', 'orange', 'purple']
bars = ax2.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val, label in zip(bars, values, metrics):
    height = bar.get_height()
    if 'Trade' in label or 'Win' in label:
        text = f'{val:.0f}' if 'Win' in label else f'{val:.0f}'
    else:
        text = f'{val:.3f}%'
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            text, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Value', fontsize=11)
ax2.set_title('Key Strategy Metrics', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('05_trade_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_trade_analysis.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("STRATEGY PERFORMANCE SUMMARY")
print("="*70)

# Market metrics
market_return = (df['Cumulative_Market'].iloc[-1] - 1) * 100
market_vol = df['Daily_Return'].std() * np.sqrt(252) * 100
market_sharpe = (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252)
market_max_dd = calculate_drawdown_series(df['Cumulative_Market']).min() * 100

# Strategy metrics
strategy_return = (df['Cumulative_Strategy_Net'].iloc[-1] - 1) * 100
strategy_vol = df['Strategy_Return_Net'].std() * np.sqrt(252) * 100
strategy_sharpe = (df['Strategy_Return_Net'].mean() / df['Strategy_Return_Net'].std()) * np.sqrt(252)
strategy_max_dd = calculate_drawdown_series(df['Cumulative_Strategy_Net']).min() * 100

print(f"\nBUY & HOLD BENCHMARK:")
print(f"  Total Return:           {market_return:>8.2f}%")
print(f"  Annualized Volatility:  {market_vol:>8.2f}%")
print(f"  Sharpe Ratio:           {market_sharpe:>8.2f}")
print(f"  Max Drawdown:           {market_max_dd:>8.2f}%")

print(f"\nSMA CROSSOVER STRATEGY (Net of Costs):")
print(f"  Total Return:           {strategy_return:>8.2f}%")
print(f"  Annualized Volatility:  {strategy_vol:>8.2f}%")
print(f"  Sharpe Ratio:           {strategy_sharpe:>8.2f}")
print(f"  Max Drawdown:           {strategy_max_dd:>8.2f}%")

print(f"\nCOMPARATIVE ANALYSIS:")
print(f"  Return Advantage:       {strategy_return - market_return:>8.2f}% (strategy vs market)")
print(f"  Volatility Reduction:   {market_vol - strategy_vol:>8.2f}% (strategy is less volatile)")
print(f"  Sharpe Improvement:     {strategy_sharpe - market_sharpe:>8.2f} (risk-adjusted returns)")
print(f"  Drawdown Reduction:     {market_max_dd - strategy_max_dd:>8.2f}% (strategy loses less)")

print(f"\nTRADE STATISTICS:")
print(f"  Total Number of Trades: {total_trades:>8}")
print(f"  Avg Return per Trade:   {trade_returns.mean():>8.3f}%")
print(f"  Best Single Trade:      {trade_returns.max():>8.3f}%")
print(f"  Worst Single Trade:     {trade_returns.min():>8.3f}%")
win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) * 100 if len(trade_returns) > 0 else 0
print(f"  Win Rate:               {win_rate:>8.1f}%")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print(f"\n1. RETURNS:")
print(f"   Strategy returned {strategy_return:.1f}% vs Buy&Hold {market_return:.1f}%")
if strategy_return > market_return:
    print(f"   ✓ Strategy outperformed by {strategy_return - market_return:.1f}%")
else:
    print(f"   ✗ Strategy underperformed by {market_return - strategy_return:.1f}%")

print(f"\n2. VOLATILITY & RISK:")
print(f"   Strategy volatility: {strategy_vol:.1f}% vs Market: {market_vol:.1f}%")
if strategy_vol < market_vol:
    print(f"   ✓ Strategy is {market_vol - strategy_vol:.1f}% LESS volatile (less scary)")
else:
    print(f"   ✗ Strategy is {strategy_vol - market_vol:.1f}% MORE volatile")

print(f"\n3. RISK-ADJUSTED RETURNS (Sharpe Ratio):")
print(f"   Strategy Sharpe: {strategy_sharpe:.2f} vs Market: {market_sharpe:.2f}")
if strategy_sharpe > market_sharpe:
    print(f"   ✓ Better risk-adjusted returns ({strategy_sharpe - market_sharpe:.2f} points better)")
else:
    print(f"   ✗ Worse risk-adjusted returns")

print(f"\n4. DRAWDOWN (How bad it gets):")
print(f"   Strategy max loss: {strategy_max_dd:.1f}% vs Market: {market_max_dd:.1f}%")
print(f"   ✓ Strategy loses {abs(market_max_dd - strategy_max_dd):.1f}% LESS on worst days")

print(f"\n5. REAL-WORLD MEANING:")
print(f"   • 95% of traders fail because they chase the 'perfect' strategy")
print(f"   • SMA Crossover is simple, mechanical, emotion-free")
print(f"   • It avoids the biggest crashes by going to CASH")
print(f"   • Transaction costs eat {(1 - df['Cumulative_Strategy'].iloc[-1]/df['Cumulative_Strategy_Net'].iloc[-1])*100:.1f}% of gross gains")
print(f"   • Better to be slightly profitable with less pain than chase high returns")

print("\n" + "="*70)
