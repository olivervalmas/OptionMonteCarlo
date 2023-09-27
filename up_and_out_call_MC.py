# Code to calculate the price of an up and out call option using Monte Carlo

np.random.seed(69)

def montecarlo_upandoutcall(t, S0, r, sigma, E, T, B):
    Vs = []
    paths = []
    M = 10**4
    for i in range(0, M):
        dt = 10**-3
        intervals = int(T/dt)
        path = [S0]
        for i in range(intervals):
            Z = np.random.normal(0,1)
            new_price = path[-1] * np.exp((r-sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)
            path.append(new_price)
        max_price = max(path)
        paths.append(path)
        val = math.exp(-r*T) * max(path[-1]-E, 0) * int(max_price<B)
        Vs.append(val)

    # print(paths)

    # for P in paths:
    #     plt.plot(P, color='blue')

    # plt.show()
    # print(Vs)

    am = np.mean(Vs)
    bm = np.std(Vs)
    print("num samples={}, am={}, 95% interval: ({}, {})".format(M, am, am-1.96*bm/(M**0.5), am+1.96*bm/(M**0.5)))
    
# montecarlo_upandoutcall(0, 5, 0.05, 0.25, 6, 1, 9)

def optimised_montecarlo_upandoutcall(t, S0, r, sigma, E, T, B):


    M = 10**5
    dt = 10**-3

    intervals = int(T/dt)
    S = np.full((M, intervals), float(S0))


    for i in range(0, intervals-1):

        if i%100==0:
            print(i)

        Z = np.random.normal(0,1,M)
        current_price = S[:, i]

        new_price = current_price * np.exp((r-sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

        S[:,i+1] = new_price

    max_prices = np.amax(S, axis=1)
    # print(max_prices)

    vals = np.exp(-r*T) * np.maximum(S[:,-1]-E, 0) * (max_prices<B).astype(int)

    plt.plot(S[0], color='blue')
    plt.show()

    am = np.mean(vals)
    bm = np.std(vals)
    print("num samples={}, am={}, 95% interval: ({}, {})".format(M, am, am-1.96*bm/(M**0.5), am+1.96*bm/(M**0.5)))

# optimised_montecarlo_upandoutcall(0, 5, 0.05, 0.25, 6, 1, 9)
