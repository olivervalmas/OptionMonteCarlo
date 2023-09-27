import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt

def call(t, S0, r, sigma, E, T):
    d1 = (np.log(S0/E) + (r + 1/2 * sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return S0*scipy.stats.norm.cdf(d1) - E * np.exp(-r*(T-t))*scipy.stats.norm.cdf(d2)

# print(call(0, 10, 0.06, 0.1, 9, 1))


def montecarlo_call(S0, r, sigma, E, T):
    # Ms = [2**n for n in range(5,25)]
    Ms = [100000]
    for M in Ms:
        Z = np.random.normal(0, 1, M)
        samples = np.maximum(S0 * np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*Z) - E, 0)
        am = np.mean(samples)
        bm = np.std(samples)
        am = np.exp(-r*T)*am
        bm = np.exp(-r*T)*bm
        print("num samples={}, am={}, 95% interval: ({}, {})".format(M, am, am-1.96*bm/(M**0.5), am+1.96*bm/(M**0.5)))

montecarlo_call(10, 0.3, 0.5, 9, 1)
