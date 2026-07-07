import numpy as np
import pandas as pd

from .synth import regime_series


def load_prices_yf(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def log_returns(prices):
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


DEMO_SEGMENTS = [
    {"kind": "gbm", "n": 240, "mu": 0.11, "sigma": 0.09, "regime": 0},
    {"kind": "gbm", "n": 180, "mu": 0.02, "sigma": 0.20, "regime": 1},
    {"kind": "merton", "n": 150, "mu": -0.32, "sigma": 0.36, "lam": 45,
     "jump_mu": -0.02, "jump_sigma": 0.03, "regime": 2},
    {"kind": "gbm", "n": 210, "mu": 0.16, "sigma": 0.11, "regime": 0},
    {"kind": "gbm", "n": 200, "mu": 0.03, "sigma": 0.23, "regime": 1},
    {"kind": "gbm", "n": 230, "mu": 0.13, "sigma": 0.08, "regime": 0},
    {"kind": "merton", "n": 140, "mu": -0.28, "sigma": 0.42, "lam": 55,
     "jump_mu": -0.018, "jump_sigma": 0.035, "regime": 2},
    {"kind": "gbm", "n": 210, "mu": 0.07, "sigma": 0.18, "regime": 1},
]


def demo_series(seed=7):
    returns, labels = regime_series(DEMO_SEGMENTS, seed=seed)
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2018-01-02", periods=returns.size)
    return dates, prices, returns, labels
