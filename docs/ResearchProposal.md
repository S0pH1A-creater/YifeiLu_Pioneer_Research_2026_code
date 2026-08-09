**Research Proposal**





**Research Question & Hypothesis**



How do Merton and Kou jump diffusion models, integrated with a GARCH-extended GBM framework, differ in modeling stock returns and assessing financial risk across post-2000 volatility regimes?



**Hypothesis**



The research hypothesizes that models incorporating both dynamic volatility and jump processes will better represent financial market behavior than traditional GBM. 

Furthermore, due to its biased jump distribution, the Kou Jump Diffusion model is expected to better at capturing extreme market movements and provide improved risk assessment compared with the symmetric Merton model.



**Background and Motivation**



Financial markets are complex systems characterized by uncertainty, volatility fluctuations, and occasional extreme events. The 2008 Global Financial Crisis demonstrated that simple traditional models may fail to adequately capture sudden market shocks. Understanding how well mathematical models capture financial market behavior is therefore important for improving the representation of uncertainty and thus for better management of risks.



Geometric Brownian Motion (GBM) has been widely used as a foundational model for stock price dynamics due to its simplicity and tractability. However, its assumptions of constant volatility and normally distributed returns limit its ability to represent real market features such as volatility clustering.



To address these limitations, this research investigates more advanced stochastic models by integrating GARCH-based volatility modeling (which makes volatility variable and improves modeling accuracy) with two different jump diffusion approaches: the symmetric Merton Jump Diffusion model and the asymmetric Kou Double Exponential Jump Diffusion model. By comparing these models under different market volatility regimes, this research aims to explore how different jump diffusion rules influence the modeling of stock returns and the evaluation of financial risks.





**Data acquisition & description** 



**Data Collection**



What real-world data:

- SPY (S&P 500 ETF): primary series for market behavior (equity prices + American options)
- AAPL, MSFT: secondary equity + options for comprehensiveness; JPM/XOM equity retained as auxiliary
- Risk-free rate: FRED 3-Month Treasury (DGS3MO)
- American option panels (SPY primary): underlying \(S_t\), strike \(K\), expiration, call/put, premium, trading date, \(r\)



Time period and market regimes:

- Crisis period: 2008–2009 (Global Financial Crisis; options from 2008)

- Normal volatility period: 2013–2014

- Late period: 2018–2019



How:

Equity prices and log returns are prepared in `research/scripts/` (`data_fetch.py`, `data_prepare.py`). SPY American options are downloaded and filtered in `options_fetch.py`, joined to \(S_t\) and \(r\), and stored under `research/data/options/`. See `research/STEP1.md` for Step 1 status and file layout.





**Methodology**



**Tools Used**



- Publicly available financial databases — historical data collection

- GitHub Copilot AI — code writing, graph generation, simulation building

- ChatGPT — reference and literature recommendations

- Manual writing — paper drafting





**Objectives**



1. Compare the performance of stochastic models with increasing complexity:

    - Geometric Brownian Motion (GBM)

    - GARCH volatility model

    - GARCH-Merton Jump Diffusion model

    - GARCH-Kou Jump Diffusion model

2. Examine how different jump processes influence the ability of models to represent daily stock return behavior and trend

3. Evaluate model performance across different post-2000 market volatility regimes, including:

    - Market crash: 2008–2009 (Global Financial Crisis)

    - Normal market conditions: 2013–2014

    - Late period: 2018–2019

4. Assess financial risk using quantitative measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR).





**Model Construction**



The following models will be implemented and compared:



1. Geometric Brownian Motion (GBM)

A baseline model assuming constant volatility and continuous normally distributed returns.



2. GARCH Model

A volatility model that captures time-varying volatility and volatility clustering in financial returns.



3. GARCH-Merton Jump Diffusion Model

A model combining GARCH volatility with normally distributed jumps to represent sudden market changes.



4. GARCH-Kou Jump Diffusion Model

A model combining GARCH volatility with an asymmetric double exponential jump process to better represent unequal upward and downward market movements.



**Return Calculation**



- Historical stock prices will first be converted into logarithmic returns

- Returns generated from each stochastic model will also be calculated, as returns provide a more meaningful measure of real-world market performance and financial uncertainty than price changes alone



**Simulation**



- Monte Carlo simulation will be used to generate possible future return scenarios from each stochastic model. These simulated returns will then be used to reconstruct corresponding stock price paths, allowing the models to represent possible future market trajectories.



**Evaluation & Conclusion**



**Evaluation**



- Comparison between GBM and GARCH-extended GBM (small proportion)
- Comparison between Merton and Kou jump diffusion model, both with underlying GARCH-extended GBM (large proportion, essay’s main focus)



results will be compared using:

- Model accuracy in reproducing price change

    - Realistic representation of historical market patterns

- Return distribution characteristics

    - Ability to capture real-world return behavior and extreme movements

- Volatility estimation

    - Effectiveness in reflecting changing market uncertainty

- VaR and CVaR measurements

    - Practical assessment of potential losses and financial uncertainty

    - VaR will measure potential losses under a selected confidence level

    - CVaR will evaluate the average loss beyond the VaR threshold.

- RMSE for error and accuracy evaluation





**Conclusion methods**



- apply bar charts (for Var adn CVaR), histograms, Monte Carlos simulated stock market price paths (for trends), RMSE (for error and accuracy evaluation), 50th/90th/95th percentile (for central tendency and anomalies avoidance)





**Evaluation & Conclusion**



- haven’t fully decided yet
- Combine review papers and academic research papers
- GitHub copilot and vscode python resources
- Search engine
- Newspaper ?

