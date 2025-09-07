import numpy as np

def payoff_long_call(price_range, strike, premium):
    return np.maximum(price_range - strike, 0) - premium

def payoff_long_put(price_range, strike, premium):
    return np.maximum(strike - price_range, 0) - premium
