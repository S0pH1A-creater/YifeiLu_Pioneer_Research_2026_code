**Research Proposal**

**Research Question & Hypothesis**

How do Heston and GARCH volatility models, integrated with a Merton jump-diﬀusion extension of the GBM framework, influence estimations and optimal stopping decisions in American option pricing across diﬀerent post-2000 volatility regimes?

**Hypothesis**

The research hypothesizes that models incorporating both jump processes and dynamic volatility will better represent financial market behavior than traditional GBM, leading to improved estimation of optimal exercise decisions for American options.

Furthermore, GARCH-based volatility models are expected to provide more accurate optimal stopping decisions than Heston stochastic volatility models in periods where historical volatility patterns and volatility clustering are dominant, as GARCH incorporates observed market behavior through parameter estimation. However, Heston models may provide advantages during highly uncertain market conditions because they explicitly model volatility as a stochastic process.

**Background and Motivation**

Financial markets are complex systems characterized by uncertainty, volatility fluctuations, and occasional extreme events. The 2008 Global Financial Crisis demonstrated that simple traditional models may fail to adequately capture sudden market shocks. Understanding how well mathematical models capture financial market behavior is therefore important for improving the representation of uncertainty and thus for better financial decision-making.

Geometric Brownian Motion (GBM) has been widely used as a foundational model for stock price dynamics due to its simplicity and tractability. its assumptions of constant volatility and normally distributed returns limit its ability to represent real market features such as volatility clustering and market shocks.

To address these limitations, this research investigates more advanced stochastic models by first

considering jump diﬀusion approaches, which introduce sudden market movements through

models such as the Merton Jump Diﬀusion model.Furthermore, this research focuses on improving volatility modeling by comparing two advanced

volatility approaches: Heston stochastic volatility model, which models volatility itself as a

stochastic process, and the GARCH-based volatility modeling, which captures time-varying

volatility and volatility clustering. By comparing these models under diﬀerent market volatility

regimes, this research aims to explore how volatility modeling influences simulated stock price

paths and optimal stopping decisions in American option pricing.

**Data acquisition & description**

**Data Collection**

Both option prices and stock price data for:

- SPY: as the primary data, representing the whole market behavior
- AAPL, MSFT: secondary / auxiliary equities (complete equity + options panels) for comprehensiveness,

not as important as SPY; JPM and XOM equity series are retained but are not part of the options focus

Option price data (specific)

- Underlying asset price S_t
- Strike price K
- Expiration date / maturity
- Option type (call/put)
- Option price (premium)
- Trading date
- Risk-free interest rate r

Time period and market regimes:

- Normal volatility period: 2013–2014
- High-volatility period: 2017–2018
- Crisis period: 2008 Global Financial Crisis

How:

stock market data from the post-2000 period will be collected from yfinance library in Python /

GitHub copilot. The dataset will include market price information, which will be transformed into

logarithmic returns for statistical modeling.

**Methodology**

**Tools Used**

- Publicly available financial databases — historical data collection- GitHub Copilot AI — code writing, graph generation, simulation building
- ChatGPT — reference and literature recommendations
- Manual writing — paper drafting

**Objectives**

1. Compare the performance of stochastic models with increasing complexity:

- Geometric Brownian Motion (GBM)
- Merton Jump Diﬀusion model
- Heston-Merton Jump Diﬀusion model
- GARCH-Merton Jump Diﬀusion model

1. Examine how diﬀerent volatility assumptions influence future stock price simulation and optimal

stopping decisions.

1. Evaluate model performance across diﬀerent post-2000 market volatility regimes, including:

- Normal market conditions: 2013-2014
- High-volatility periods: 2017-2018
- Market crash: The 2008 Global Financial Crisis

1. Apply simulated stock price paths to American option pricing and determine optimal stopping

decisions.

1. Compare how diﬀerent stochastic models influence estimated option exercise timing and

estimated expected option value.

**Model Construction**

The following models will be implemented and compared:

1. Geometric Brownian Motion (GBM)

A baseline model assuming constant volatility and continuous normally distributed returns.

1. Merton Jump Diﬀusion Model

A model incorporating sudden random price jumps to represent extreme market movements.

1. Heston-Merton Jump Diﬀusion Model

A model combining the Heston stochastic volatility model with the Merton Jump Diﬀusion

process, allowing both uncertain volatility changes and sudden extreme price movements to be

represented.

1. GARCH-Merton Jump Diﬀusion ModelA model combining GARCH-based time-varying volatility with the Merton Jump Diﬀusion process,

capturing both volatility clustering and sudden market shocks.

**Return Calculation**

- Historical stock prices will first be converted into logarithmic returns
- Returns generated from each stochastic model will also be calculated, as returns provide a more

meaningful measure of real-world market performance and financial uncertainty than price

changes alone

**Simulation**

- Monte Carlo simulation will be used to generate possible future return scenarios from each

stochastic model.

- These simulated returns will then be reconstructed into corresponding stock price paths,

allowing the models to represent possible future market trajectories.

- The simulated stock price paths will be combined with American option information to calculate

possible future payoﬀs and determine optimal stopping decisions.

**Optimal Stopping Method (at each step/each day, we…)**

1. Calculate the immediate exercise payoﬀ: max (St-k, 0) for an American call option.
2. Use Monte Carlo simulations to estimate the expected payoﬀ from waiting.
3. Compare: immediate exercise value & Expected value of continuing to hold the option
4. Select the decision with the higher expected value.

This process is repeated across diﬀerent time periods and stochastic models to determine how

volatility assumptions influence optimal exercise timing.

**Evaluation & Conclusion**

**Evaluation**

- Comparison between GBM and Merton Jump Diﬀusion model (small proportion)
- Comparison between Heston-Merton and GARCH-Merton Jump Diﬀusion models (large

proportion, essay’s main focus)

Result comparison using:

- Model accuracy in reproducing historical market patterns
- Realistic representation of stock price movements and market behavior- Return distribution characteristics for calculation of future stock price
- Ability to capture real-world return behavior and extreme movements
- Volatility estimation
- Eﬀectiveness in reflecting changing market uncertainty
- American option pricing prediction
- Exptected option value generated by each stochastic model
- Optimal stopping performance
- Estimated optimal exercise timing and diﬀerences between models under diﬀerent volatility

regimes

- Monte Carlo visualization

Quantitative evaluation:

- RMSE for error and accuracy evaluation
- Comparison of simulated and historical market behavior

**Conclusion methods**

- Bar charts (for comparing option values and model results)
- Histograms (for return distribution comparison)
- Monte Carlo simulated stock market price paths (for trends)
- RMSE (for error and accuracy evaluation)
- 50th/90th/95th percentile (for central tendency and anomaly analysis)

**Reference and resource**

- haven’t fully decided yet
- Combine review papers and academic research papers
- GitHub copilot and vscode python resources
- Search engine
- Newspaper ?

