#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:36:01 2021

@author: davidwalch
"""

import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Define Assets

stocks = ["^GSPC"]
crypto = ["BTC-USD"]

tickers = ["^GSPC", "BTC-USD"]

def get_data(tickers, start_date, end_date):
    
    # download daily price data for each of the assets in the portfolio
    df = pdr.get_data_yahoo(tickers, start=start_date, end = end_date)['Adj Close']
    df.sort_index(inplace = True)
    
    return df

# Download Data in defined timeframe

stock_prices = get_data(stocks, '01/01/2015', '01/01/2021' )

crypto_prices = get_data(crypto, '01/01/2015', '01/01/2021' )

# Merge Data into one Dataframe

data_prices = pd.concat([stock_prices, crypto_prices], axis=1)

data_prices = data_prices.fillna(method ='ffill')
data_prices = data_prices.fillna(method ='bfill')

# Calculate daily returns

daily_returns = data_prices.pct_change()

# Calculate mean daily returns

mean_daily_returns = daily_returns.mean()

# Calculate Covarinace Matrix

cov_matrix = daily_returns.cov()

# Define Monte Carlo Simulation

def create_results_dataframe(tickers ,number_portfolios, mean_daily_returns, cov_matrix):
    results_temp = np.zeros((4 + 2 - 1 , number_portfolios))

    for i in range(number_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(2))
        
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 365 * 6
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365 * 6)

        # store results in results array
        results_temp[0, i] = portfolio_return
        results_temp[1, i] = portfolio_std_dev
        
        # store Sharpe Ratio (return / volatility)
        results_temp[2, i] = (results_temp[0, i]) / results_temp[1, i]
        
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_temp[j + 3, i] = weights[j]

    # convert results array to Pandas DataFrame
    results_df = pd.DataFrame(results_temp.T, columns=['ret', 'stdev', 'sharpe', tickers[0], 
                                                       tickers[1]])
    
    return results_df

crypto_results = create_results_dataframe(tickers, 10000, mean_daily_returns, cov_matrix)

# Locate Portfolio with highest Sharpe Ratio

def max_sharpe_ratio(results_df):
    return results_df.iloc[results_df['sharpe'].idxmax()]

max_sharpe_portfolio = max_sharpe_ratio(crypto_results)
print("-"*80)
print("Maximum Sharpe Ratio Portfolio Allocation\n")
print(max_sharpe_portfolio)
print("-"*80)

# Locate Portfolio with loewst volatility

def min_volatility(results_df):
    return results_df.iloc[results_df['stdev'].idxmin()]

min_vol_portfolio = min_volatility(crypto_results)
print("-"*80)
print("Minimum Volatility Portfolio Allocation\n")
print(min_vol_portfolio)
print("-"*80)

# Plot "efficient frontier"

def plot_graph(results_df, max_sharpe_port, min_vol_port):
    ax = results_df.plot(kind= 'scatter', x = 'stdev', y='ret', s = 30, 
                         c=results_df.sharpe, cmap='viridis', figsize=(10,6))
    ax.grid(False, color='w', linestyle='-', linewidth=1)
    ax.set_facecolor('4444')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Returns')
    ax.tick_params(labelsize = 14)

    # plot red dot to highlight position of portfolio with highest Sharpe Ratio
    ax.scatter(max_sharpe_port[1], max_sharpe_port[0], color='r', s=50)
    # # plot red dot to highlight position of minimum variance portfolio
    ax.scatter(min_vol_port[1], min_vol_port[0], color='r', s=50)
    
plot_graph(crypto_results, max_sharpe_portfolio, min_vol_portfolio)
plt.show()