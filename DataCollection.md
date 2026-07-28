### **1. Stock market data**

For:

- **SPY** (primary)
- **AAPL**, **MSFT** (secondary)
- **JPM**, **XOM** (auxiliary — retained)

Download:

- Date
- Adjusted closing price

(Used to calculate log returns and estimate GBM/GARCH/Heston/Merton parameters.)

**Status:** ✓ SPY, AAPL, MSFT, JPM, XOM in `research/data/equity/` (prices + log returns + regime stats).

---

### **2. American call option data**

Focus set (complete):

- **SPY** (primary)
- **AAPL**, **MSFT** (secondary)

Auxiliary (equity kept; options not required):

- **JPM**, **XOM**

Download:

- Trading date
- Underlying stock price S_t
- Strike price K
- Expiration date
- Option price (premium)
- Option type (call)

(Used for American option pricing and optimal stopping.)

**Status:** ✓ SPY + AAPL + MSFT American option panels (all three regimes). JPM / XOM left equity-only (no free Cobweb dumps; CDN offline).

---

### **3. Risk-free interest rate data**

Download:

- U.S. Treasury risk-free rate

(Used to discount future option payoffs.)

**Status:** ✓ FRED DGS3MO in `research/data/rates/risk_free_dgs3mo.csv`.

### **Stock price data**

Download:

- **Post-2000 period (2000–present)**

Purpose:

- Enough data to estimate model parameters (GARCH/Heston/Merton).
- Then select specific regimes for comparison.

### **Main evaluation periods:**

1. **Crisis period**
  - 2008 Global Financial Crisis
2. **Normal volatility period**
  - 2013–2014
3. **High-volatility period**
  - 2017–2018

### **Option data**

Use:

- The same three periods:
  - 2008
  - 2013–2014
  - 2017–2018
