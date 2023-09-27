# Find the 95% confidence for an European up-and-out-call option C(t, S_0, E, r, sigma, B, T) using
# the standard Monte Carlo method and the corresponding antithetic method
# using variance reduction technique. Compare the widths of the corresponding
# intervals.
# The parameters given are
# t=0, initial stock price S_0=5, strike price E= 6, interest rate r=0.05,
# volatility sigma = 0.25, barrier B = 9, and maturity time T = 1.
# the time interval [0, T] is divided into N = 10^3 equal intervals.
# So C(t, S_0, E, r, sigma, B, T) = max (S(T) -B)1_{max_{[0, T]} S(t) >= B}.
# Number of simulations M= 10^4.
import scipy.stats as s
import math
import numpy as np
import statistics
import pandas as pd
# fixing a seed to produce the same sequence of random numbers
np.random.seed(119)
M=[ 10**4 ]


# The standard Monte Carlo option
def MC_up_and_out_call_option_C(t, S_0, E, r, sigma, B, T):
    V=[]
    sample_mean = []
    confidence_interval = []
    for j in range(0,len(M)):
        for i in range(0, M[j]):
            S=[S_0]
            D_t=10**-3
            N= int(T/D_t)
            for k in range(0, N):
                xi= np.random.normal(0,1)
                stock_price_updated = S[k]* math.exp((r - 0.5* sigma ** 2) * D_t
                                                     + sigma * math.sqrt(D_t) * xi)
                S.append(stock_price_updated)
            max_S=max(S)
            value = math.exp(-r * T) * max(S[N] - E, 0)*int(max_S<B)
            V.append(value)
        sample_mean_M = statistics.mean(V)
        sample_mean.append(sample_mean_M)
        sample_sd_M= statistics.stdev(V)
        lower_bound = sample_mean_M - 1.96 / math.sqrt(M[j]) * sample_sd_M
        upper_bound = sample_mean_M + 1.96 / math.sqrt(M[j]) * sample_sd_M
        confidence_interval_component = [lower_bound, upper_bound]
        confidence_interval.append(confidence_interval_component)
    return sample_mean, confidence_interval

print(MC_up_and_out_call_option_C(0, 5, 6, 0.05, 0.25, 9, 1))