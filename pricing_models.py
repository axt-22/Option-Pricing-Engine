import math
import numpy as np
from datetime import date, timedelta
from abc import ABC, abstractmethod
from scipy.stats import norm

# Option class
class Option:
    def __init__(self, option_type, style, strike_price, underlying, n_day):
        self.option_type = option_type
        self.style = style
        self.strike_price = strike_price
        self.underlying = underlying
        self.expiry = date.today() + timedelta(days=n_day)

# Market Data class
class MarketData:
    def __init__(self, spot_price, volatility, risk_free, dividend):
        self.spot_price = spot_price
        self.volatility = volatility
        self.risk_free = risk_free
        self.dividend = dividend

# Abstract base class for pricing models
class PricingModel(ABC):
    def __init__(self, option, marketdata):
        self.option = option
        self.marketdata = marketdata

    @abstractmethod
    def price(self):
        pass

    @abstractmethod
    def greeks(self):
        pass

# Black-Scholes Model
def calculations_BlackScholes(self):
    S = self.marketdata.spot_price
    K = self.option.strike_price
    T = (self.option.expiry - date.today()).days / 365
    sigma = self.marketdata.volatility
    r = self.marketdata.risk_free
    q = self.marketdata.dividend
    option_type = self.option.option_type

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S, K, T, sigma, r, q, option_type, d1, d2

def N_deriv(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

class BlackScholes(PricingModel):
    def price(self):
        S, K, T, sigma, r, q, option_type, d1, d2 = calculations_BlackScholes(self)
        if option_type == 'call':
            return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

    def greeks(self):
        S, K, T, sigma, r, q, option_type, d1, d2 = calculations_BlackScholes(self)
        if option_type == 'call':
            delta = math.exp(-q * T) * norm.cdf(d1)
            theta = (-S * sigma * math.exp(-q * T) * N_deriv(d1)) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2) + q * S * math.exp(-q * T) * norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        else:
            delta = -math.exp(-q * T) * norm.cdf(-d1)
            theta = (-S * sigma * math.exp(-q * T) * N_deriv(d1)) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2) - q * S * math.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

        gamma = math.exp(-q * T) * N_deriv(d1) / (S * sigma * math.sqrt(T))
        vega = S * math.exp(-q * T) * N_deriv(d1) * math.sqrt(T)

        return delta, gamma, vega, theta, rho

# Binomial Model
def calculations_Binomial(self):
    S = self.marketdata.spot_price
    K = self.option.strike_price
    T = (self.option.expiry - date.today()).days / 365
    sigma = self.marketdata.volatility
    r = self.marketdata.risk_free
    q = self.marketdata.dividend
    option_type = self.option.option_type
    return S, K, T, sigma, r, q, option_type

def delta_calc(x, y, option_tree, price_tree):
    return (option_tree[x][y] - option_tree[x][y - 1]) / (price_tree[x][y] - price_tree[x][y - 1])

class Binomial(PricingModel):
    def price(self, N):
        S, K, T, sigma, r, q, option_type = calculations_Binomial(self)
        self.N = N
        delta_t = T / N
        u = math.exp(sigma * math.sqrt(delta_t))
        d = 1 / u
        p = (math.exp((r - q) * delta_t) - d) / (u - d)

        price_tree = [[0.0 for _ in range(i + 1)] for i in range(N + 1)]
        for i in range(N + 1):
            for j in range(i + 1):
                price_tree[i][j] = S * (u ** j) * (d ** (i - j))

        option_tree = [[0.0 for _ in range(i + 1)] for i in range(N + 1)]
        if option_type == 'call':
            for j in range(N + 1):
                option_tree[N][j] = max(price_tree[N][j] - K, 0)
        else:
            for j in range(N + 1):
                option_tree[N][j] = max(K - price_tree[N][j], 0)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                expected = math.exp(-r * delta_t) * (p * option_tree[i + 1][j + 1] + (1 - p) * option_tree[i + 1][j])
                intrinsic = max(price_tree[i][j] - K, 0) if option_type == 'call' else max(K - price_tree[i][j], 0)
                option_tree[i][j] = max(expected, intrinsic)

        return option_tree, price_tree

    def greeks(self, option_tree, price_tree):
        S, K, T, sigma, r, q, option_type = calculations_Binomial(self)
        delta = delta_calc(1, 1, option_tree, price_tree)
        gamma = (delta_calc(2, 2, option_tree, price_tree) - delta_calc(1, 1, option_tree, price_tree)) / ((price_tree[2][2] - price_tree[2][0]) / 2)

        # Vega
        sigma_init = self.marketdata.volatility
        self.marketdata.volatility = sigma_init + 0.01
        tree_up, _ = self.price(self.N)
        V_up = tree_up[0][0]

        self.marketdata.volatility = sigma_init - 0.01
        tree_down, _ = self.price(self.N)
        V_down = tree_down[0][0]

        self.marketdata.volatility = sigma_init
        vega = (V_up - V_down) / 0.02

        # Rho
        r_init = self.marketdata.risk_free
        self.marketdata.risk_free = r_init + 0.01
        tree_up, _ = self.price(self.N)
        R_up = tree_up[0][0]

        self.marketdata.risk_free = r_init - 0.01
        tree_down, _ = self.price(self.N)
        R_down = tree_down[0][0]

        self.marketdata.risk_free = r_init
        rho = (R_up - R_down) / 0.02

        # Theta
        T_old = option_tree[0][0]
        t_init = self.option.expiry
        self.option.expiry -= timedelta(days=1)
        tree_new, _ = self.price(self.N)
        T_new = tree_new[0][0]
        self.option.expiry = t_init
        theta = (T_new - T_old) / (1 / 365)

        return delta, gamma, vega, theta, rho

# Monte Carlo Model
class MonteCarlo(PricingModel):
    def price(self, sim, time_step):
        np.random.seed(42)
        S, K, T, sigma, r, q, option_type, d1, d2 = calculations_BlackScholes(self)
        self.sim = sim
        self.time_step = time_step
        delta_t = T / time_step

        payoffs = []
        for _ in range(sim):
            S_t = S
            for _ in range(time_step):
                S_t *= math.exp((r - q - 0.5 * sigma ** 2) * delta_t + sigma * math.sqrt(delta_t) * np.random.normal())
            payoff = max(S_t - K, 0) if option_type == 'call' else max(K - S_t, 0)
            payoffs.append(payoff)

        option_price = np.mean(payoffs) * math.exp(-r * T)
        return option_price

    def greeks(self, epsilon=0.01):
        S_init = self.marketdata.spot_price
        self.marketdata.spot_price = S_init + epsilon
        S_up = self.price(self.sim, self.time_step)
        self.marketdata.spot_price = S_init - epsilon
        S_down = self.price(self.sim, self.time_step)
        self.marketdata.spot_price = S_init
        S_base = self.price(self.sim, self.time_step)

        delta = (S_up - S_down) / (2 * epsilon)
        gamma = (S_up + S_down - 2 * S_base) / (epsilon ** 2)

        sigma_init = self.marketdata.volatility
        self.marketdata.volatility = sigma_init + epsilon
        V_up = self.price(self.sim, self.time_step)
        self.marketdata.volatility = sigma_init - epsilon
        V_down = self.price(self.sim, self.time_step)
        self.marketdata.volatility = sigma_init
        vega = (V_up - V_down) / (2 * epsilon)

        r_init = self.marketdata.risk_free
        self.marketdata.risk_free = r_init + epsilon
        R_up = self.price(self.sim, self.time_step)
        self.marketdata.risk_free = r_init - epsilon
        R_down = self.price(self.sim, self.time_step)
        self.marketdata.risk_free = r_init
        rho = (R_up - R_down) / (2 * epsilon)

        T_old = S_base
        t_init = self.option.expiry
        self.option.expiry -= timedelta(days=1)
        T_new = self.price(self.sim, self.time_step)
        self.option.expiry = t_init
        theta = (T_new - T_old) / (1 / 365)

        return delta, gamma, vega, theta, rho
