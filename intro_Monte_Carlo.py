
import numpy as np
M = 10000000
samples = np.exp(np.random.normal(0, 1, M))
am = np.mean(samples)
bm = np.var(samples, ddof=1)

print("am={}".format(am))
print("97% confidence interval: {}<=am<={}".format(am-2.17*bm/(M**0.5), am+2.17*bm/(M**0.5)))
print("analytical sol: {}".format(np.e**0.5))
