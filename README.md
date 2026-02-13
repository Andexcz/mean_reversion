Mean Reversion: A Fat-Tail Approach
Project Overview
This project is a backtesting engine to explore the efficacy of Student's t-distributions (Fat Tails) in modeling financial returns.

Unlike standard models that assume stock returns follow a Normal Distribution, this project utilizes a t-distribution with low degrees of freedom (df=3) to more "accurately" capture "Black Swan" events and extreme market deviations.

The Experiment: I applied this mean-reversion logic to the S&P 500 (SPY).

Note: I anticipated that a pure Mean Reversion strategy would underperform on a strongly trending asset like the S&P 500. The goal was not to build a "money printer," but to stress-test the statistical signals against a hostile market regime.

🧠 The Strategy Logic
Instead of using arbitrary "Z-Score > 2" thresholds, the strategy trades based on Probability:
    Calculate Rolling Statistics: 20-day Mean and Volatility.
    Compute t-Statistic: Standardize the current price relative to recent volatility.
    Determine Probability: Feed the t-statistic into a Student's t-distribution (df=3) CDF.
    
    Trade Logic:Long Entry: 
    Price is in the bottom 5% of probable outcomes ($p < 0.05$).
    Short Entry: Price is in the top 5% of probable outcomes ($p > 0.95$).
    Exit: Price returns to the median probability ($p \approx 0.5$).
    Stop Loss: Probability hits extreme tail limits ($p < 0.005$).

Disclaimer: I was too sick to create a readme file on my own, so like 80% of what you just read was AI generated. I'm sorry :(
    