import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from scipy import optimize


def fetchData(tickers, start, end):
    dailyReturns = pd.DataFrame(columns=tickers)

    plt.figure(figsize=(15, 12))
    plt.suptitle("Closing Prices")

    for n, ticker in enumerate(tickers):
        data = yf.download(ticker, start=start, end=end)
        dailyReturns[ticker] = data['Adj Close']

        ax = plt.subplot(len(tickers), round(len(tickers) / 2), n + 1)

        dailyReturns[ticker].plot()

        ax.set_title(ticker)
        ax.set_xlabel("")

    plt.show()
    return dailyReturns


def computeGraphParameters(dailyReturn):
    logReturns = np.log(1 + dailyReturn.pct_change())
    annualReturns = logReturns.mean() * 252
    annualReturns.plot(kind='bar')
    plt.title('Expected Annual Return')
    plt.show()

    volatility = logReturns.std() * np.sqrt(252)
    volatility.plot(kind='bar')
    plt.title('Standard Deviation (Volatility)')
    plt.show()

    cov_matrix = logReturns.cov()
    corr_matrix = logReturns.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    fig, ax = plt.subplots()
    ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')

    for i, ticker in enumerate(tickers):
        ax.scatter(x=volatility[ticker], y=annualReturns[ticker], alpha=0.5)
        ax.annotate(ticker, (volatility[ticker], annualReturns[ticker]))

    plt.show()

    return logReturns, annualReturns, volatility, cov_matrix, corr_matrix

def monteCarloSim(numSimulations, logReturns, volatility, annualReturns):
    port_weights = np.zeros(shape=(numSimulations, len(tickers)))
    port_volatility = np.zeros(numSimulations)
    port_sr = np.zeros(numSimulations)
    port_return = np.zeros(numSimulations)

    # weights = pd.DataFrame(columns=tickers)


    for i in range(numSimulations):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        port_weights[i, :] = weights


        exp_ret = logReturns.mean().dot(weights) * 252
        port_return[i] = exp_ret

        exp_vol = np.sqrt(weights.T.dot(252 * logReturns.cov().dot(weights)))
        port_volatility[i] = exp_vol

        # Sharpe ratio
        sr = exp_ret / exp_vol
        port_sr[i] = sr

    # Index of max Sharpe Ratio
    max_sr = port_sr.max()
    ind = port_sr.argmax()
    # Return and Volatility at Max SR
    max_sr_ret = port_return[ind]
    max_sr_vol = port_volatility[ind]

    plt.figure(figsize=(10, 8))
    plt.scatter(port_volatility, port_return, c=port_sr, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    plt.title('Efficient Frontier', fontsize=15)
    plt.scatter(max_sr_vol, max_sr_ret, c='blue', s=150, edgecolors='red', marker='o', label='Max \
    Sharpe ratio Portfolio')

    for i, ticker in enumerate(tickers):
        plt.scatter(x=volatility[ticker], y=annualReturns[ticker], alpha=0.5)
        plt.annotate(ticker, (volatility[ticker], annualReturns[ticker]))

    plt.legend()

    plt.show()




tickers = ['VGT', 'MSFT', 'VHGEX']
numSimulations = 50000

dailyReturn = fetchData(tickers, start='2022-07-15', end=date.today())
logReturns, annualReturns, volatility, cov_matrix, corr_matrix = computeGraphParameters(dailyReturn)
monteCarloSim(numSimulations, logReturns, volatility, annualReturns)