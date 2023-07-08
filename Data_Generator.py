import numpy as np
from scipy.stats import norm
from random import randint, uniform
import pandas as pd

# Parameters
numberOfSamples = 100000
S = []
K = []
t = []
r = []
sigma = []
bS = []

# continuously compounded dividend yield (q) = 0

def blackScholesEquation(S, K, t, r, sigma, q=0):

    d1 = (np.log(S / K) + t * (r - q + (sigma**2 / 2)))/(sigma * np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)

    # Black-Scholes Equation
    # For increased efficiency, given q=0, e^-qt = 1
    callPrice = S * norm.cdf(d1) - K * np.exp(-r * t)*norm.cdf(d2)

    # Delta
    delta = norm.cdf(d1)

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))

    # Theta

    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    theta = theta / 365

    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(t)
    vega = vega * 0.01

    # Rho
    rho = K * t * np.exp(-r * t) * norm.cdf(d2)
    rho = rho * 0.01

    return S, K, t, r, sigma, callPrice, delta, gamma, theta, vega, rho

def createData(numberOfSamples):
    for i in range(numberOfSamples):
        # Stock prices between $10 and $500
        S.append(randint(10, 500))
        # Keeping the strike price within the range of the Stock price
        K.append(S[i]+uniform(-2, 2))
        # 1/365 denotes 1 day, 3 denotes 3 years
        t.append(uniform(1/365, 3))
        # Risk free rate as a percentage
        r.append(uniform(0.01, 0.03))
        # Volatility as a percentage
        sigma.append(uniform(0.05, 0.9))
        # Calculate Call Price
        bS.append(blackScholesEquation(S[i], K[i], t[i], r[i], sigma[i]))

    option_df = pd.DataFrame(bS, columns=['Stock Price', 'Strike Price', 'Time to Maturity', 'Risk Free Rate', 'Implied Volatility', 'Call Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'])
    option_df.to_csv('optionsData.csv', index=False)

createData(numberOfSamples)

















