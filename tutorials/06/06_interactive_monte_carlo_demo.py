"""
Interactive Monte Carlo Simulations Demo
==========================================
Based on 06_Interactive_Monte_Carlo.ipynb

This demo allows you to:
1. Enter stock parameters (S0, mu, sigma, T)
2. Choose number of paths and comparison scenarios
3. Generate comparable paths and final distributions
4. Save graphs to disk for analysis

Concepts:
- Compare different parameter settings side-by-side
- Visualize effect of volatility, drift, and time horizon
- Compare normal vs fat-tailed distributions
"""


from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time

print("=" * 80)
print("INTERACTIVE MONTE CARLO SIMULATIONS DEMO")
print("=" * 80)

# ============================================================================
# STEP 1: Get Stock Data or Use Default
# ============================================================================
print("\n[SETUP] Fetching real stock data to estimate parameters...")

ticker = "SPY"  # Use SPY as it's more stable
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)

data = None
max_retries = 2
for attempt in range(max_retries):
    try:
        print(f"Downloading {ticker} data (attempt {attempt+1}/{max_retries})...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            print(f"✓ Downloaded {len(data)} trading days")
            break
    except Exception as e:
        print(f"✗ Attempt {attempt+1} failed, generating synthetic data...")
        time.sleep(2)

if data is None or len(data) == 0:
    print("Using realistic synthetic market data...")
    n_days = 500
    drift = 0.0002
    volatility = 0.012
    price_init = 450
    returns_synthetic = np.random.normal(drift, volatility, n_days)
    close_prices = price_init * np.exp(np.cumsum(returns_synthetic))
    import pandas as pd
    dates = pd.date_range(end=end_date, periods=n_days, freq='B')
    data = pd.DataFrame({'Adj Close': close_prices}, index=dates)

returns = data['Adj Close'].pct_change().dropna()
mu_daily = returns.mean()
sigma_daily = returns.std()
mu_annual = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)
current_price = data['Adj Close'].iloc[-1]

print(f"\nEstimated from {ticker}:")
print(f"  Current Price: ${current_price:.2f}")
print(f"  Annual Drift (μ): {mu_annual:.4f}")
print(f"  Annual Volatility (σ): {sigma_annual:.4f}")

# ============================================================================
# STEP 2: Interactive Parameter Input
# ============================================================================
print("\n" + "=" * 80)
print("ENTER SIMULATION PARAMETERS")
print("=" * 80)

def get_float_input(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if user_input == "":
                return default
            return float(user_input)
        except ValueError:
            print("  ✗ Invalid input. Please enter a number.")

def get_int_input(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if user_input == "":
                return default
            return int(user_input)
        except ValueError:
            print("  ✗ Invalid input. Please enter an integer.")

def get_choice_input(prompt, options, default_idx=0):
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = ">" if i == default_idx else " "
        print(f"  {marker} [{i+1}] {opt}")
    
    while True:
        try:
            user_input = input(f"Choose (default: {default_idx+1}): ").strip()
            if user_input == "":
                return options[default_idx]
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            print("  ✗ Invalid choice. Please try again.")
        except ValueError:
            print("  ✗ Invalid input. Please enter a number.")

S0 = get_float_input("Initial Stock Price (S0)", current_price)
mu = get_float_input("Annual Drift Rate (μ)", mu_annual)
sigma = get_float_input("Annual Volatility (σ)", sigma_annual)
T = get_float_input("Time Horizon in Years (T)", 1.0)
N_paths = get_int_input("Number of Paths", 500)

comparison_mode = get_choice_input(
    "What would you like to compare?",
    ["Single Scenario", "Normal vs Fat-Tailed", "Different Volatilities", "Different Time Horizons"],
    0
)

# ============================================================================
# STEP 3: Generate Monte Carlo Paths Based on Mode
# ============================================================================
print("\n[GENERATING] Computing Monte Carlo simulations...")

N_steps = int(252 * T)
dt = T / N_steps

def generate_paths(S0, mu, sigma, T, N_paths, distribution='Normal'):
    """Generate Monte Carlo paths with specified distribution"""
    N_steps = int(252 * T)
    dt = T / N_steps
    
    paths = np.zeros((N_steps, N_paths))
    paths[0] = S0
    
    for t in range(1, N_steps):
        if distribution == 'Normal':
            Z = np.random.standard_normal(N_paths)
        elif distribution == 'Fat-Tailed':
            Z = np.random.standard_t(3, size=N_paths)
        else:
            Z = np.random.standard_normal(N_paths)
        
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths

# ============================================================================
# Generate scenarios based on user choice
# ============================================================================
scenarios = {}

if comparison_mode == "Single Scenario":
    print("Generating single scenario...")
    paths = generate_paths(S0, mu, sigma, T, N_paths, 'Normal')
    scenarios['Normal Distribution'] = paths
    
elif comparison_mode == "Normal vs Fat-Tailed":
    print("Generating normal and fat-tailed scenarios...")
    paths_normal = generate_paths(S0, mu, sigma, T, N_paths, 'Normal')
    paths_fattail = generate_paths(S0, mu, sigma, T, N_paths, 'Fat-Tailed')
    scenarios['Normal Distribution'] = paths_normal
    scenarios['Fat-Tailed Distribution'] = paths_fattail
    
elif comparison_mode == "Different Volatilities":
    print("Generating paths with different volatilities...")
    sigma_low = sigma * 0.5
    sigma_mid = sigma
    sigma_high = sigma * 1.5
    
    paths_low = generate_paths(S0, mu, sigma_low, T, N_paths, 'Normal')
    paths_mid = generate_paths(S0, mu, sigma_mid, T, N_paths, 'Normal')
    paths_high = generate_paths(S0, mu, sigma_high, T, N_paths, 'Normal')
    
    scenarios[f'σ = {sigma_low:.4f} (Low)'] = paths_low
    scenarios[f'σ = {sigma_mid:.4f} (Mid)'] = paths_mid
    scenarios[f'σ = {sigma_high:.4f} (High)'] = paths_high
    
elif comparison_mode == "Different Time Horizons":
    print("Generating paths with different time horizons...")
    T_short = T * 0.5
    T_mid = T
    T_long = T * 2.0
    
    paths_short = generate_paths(S0, mu, sigma, T_short, N_paths, 'Normal')
    paths_mid = generate_paths(S0, mu, sigma, T_mid, N_paths, 'Normal')
    paths_long = generate_paths(S0, mu, sigma, T_long, N_paths, 'Normal')
    
    scenarios[f'T = {T_short:.1f} Years'] = paths_short
    scenarios[f'T = {T_mid:.1f} Years'] = paths_mid
    scenarios[f'T = {T_long:.1f} Years'] = paths_long

# ============================================================================
# STEP 4: Create Comparison Graphs
# ============================================================================
print("Creating visualization graphs...")

if len(scenarios) == 1:
    # Single scenario: show paths and distribution
    scenario_name = list(scenarios.keys())[0]
    paths = scenarios[scenario_name]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Paths plot
    time_axis = np.linspace(0, T, N_steps)
    for i in range(min(N_paths, 100)):
        ax1.plot(time_axis, paths[:, i], linewidth=0.8, alpha=0.3, color='steelblue')
    
    ax1.set_title(f'{scenario_name}: {min(N_paths, 100)} Sample Paths', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (Years)', fontsize=11)
    ax1.set_ylabel('Stock Price ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(S0, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Initial Price: ${S0:.2f}')
    ax1.legend()
    
    # Distribution plot
    ST = paths[-1, :]
    ax2.hist(ST, bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    ax2.axvline(ST.mean(), color='red', linestyle='-', linewidth=2.5, label=f'Mean: ${ST.mean():.2f}')
    ax2.axvline(np.median(ST), color='orange', linestyle='--', linewidth=2.5, label=f'Median: ${np.median(ST):.2f}')
    
    ax2.set_title(f'Final Price Distribution ({scenario_name})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Final Stock Price ($)', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = str(OUTPUT_DIR / 'comparison_single_scenario.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_single_scenario.png")
    plt.close()

else:
    # Multiple scenarios: compare paths side-by-side
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6*n_scenarios, 5))
    
    if n_scenarios == 1:
        axes = [axes]
    
    for idx, (scenario_name, paths) in enumerate(scenarios.items()):
        ax = axes[idx]
        time_axis = np.linspace(0, T, paths.shape[0])
        
        # Plot sample paths
        for i in range(min(N_paths, 100)):
            ax.plot(time_axis, paths[:, i], linewidth=0.8, alpha=0.3, color='steelblue')
        
        ax.set_title(f'{scenario_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (Years)', fontsize=10)
        ax.set_ylabel('Stock Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(S0, color='green', linestyle=':', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    filename = str(OUTPUT_DIR / 'comparison_paths.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_paths.png")
    plt.close()
    
    # Create distribution comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['steelblue', 'coral', 'green', 'purple']
    
    for idx, (scenario_name, paths) in enumerate(scenarios.items()):
        ST = paths[-1, :]
        ax.hist(ST, bins=50, alpha=0.5, label=scenario_name, color=colors[idx % len(colors)], density=True, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Comparison of Final Price Distributions', fontsize=13, fontweight='bold')
    ax.set_xlabel('Final Stock Price ($)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = str(OUTPUT_DIR / 'comparison_distributions.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_distributions.png")
    plt.close()
    
    # Create percentile comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table_data = [['Scenario', 'Mean', 'Median', '5th %ile', '95th %ile', 'Std Dev']]
    
    for scenario_name, paths in scenarios.items():
        ST = paths[-1, :]
        table_data.append([
            scenario_name,
            f'${ST.mean():.2f}',
            f'${np.median(ST):.2f}',
            f'${np.percentile(ST, 5):.2f}',
            f'${np.percentile(ST, 95):.2f}',
            f'${ST.std():.2f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Statistical Summary of Scenarios', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    filename = str(OUTPUT_DIR / 'comparison_statistics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_statistics.png")
    plt.close()

# ============================================================================
# STEP 5: Print Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("SIMULATION RESULTS SUMMARY")
print("=" * 80)

print(f"\nParameters Used:")
print(f"  Initial Price (S0): ${S0:.2f}")
print(f"  Annual Drift (μ): {mu:.4f}")
print(f"  Annual Volatility (σ): {sigma:.4f}")
print(f"  Time Horizon (T): {T:.1f} years")
print(f"  Number of Paths: {N_paths}")
print(f"  Comparison Mode: {comparison_mode}")

print(f"\nResults:")
for scenario_name, paths in scenarios.items():
    ST = paths[-1, :]
    print(f"\n  {scenario_name}:")
    print(f"    Mean Price: ${ST.mean():.2f}")
    print(f"    Median Price: ${np.median(ST):.2f}")
    print(f"    Std Dev: ${ST.std():.2f}")
    print(f"    Min: ${ST.min():.2f}, Max: ${ST.max():.2f}")
    print(f"    5th %ile: ${np.percentile(ST, 5):.2f}")
    print(f"    95th %ile: ${np.percentile(ST, 95):.2f}")

print("\n" + "=" * 80)
print("✓ Interactive Monte Carlo Demo Complete!")
print("All graphs have been saved to the 06/ folder.")
print("=" * 80)
