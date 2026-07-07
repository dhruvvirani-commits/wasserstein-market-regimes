import numpy as np

from src.synth import gbm_returns, merton_returns, regime_series
from src.data import demo_series, log_returns


def test_gbm_shape_and_reproducible():
    a = gbm_returns(500, 0.1, 0.2, seed=0)
    b = gbm_returns(500, 0.1, 0.2, seed=0)
    assert a.shape == (500,)
    assert np.allclose(a, b)


def test_merton_has_fatter_tails_than_gbm():
    g = gbm_returns(4000, 0.0, 0.2, seed=1)
    m = merton_returns(4000, 0.0, 0.2, 60, -0.02, 0.04, seed=1)
    assert m.std() > g.std()


def test_regime_series_lengths_align():
    segs = [
        {"kind": "gbm", "n": 100, "mu": 0.1, "sigma": 0.1, "regime": 0},
        {"kind": "gbm", "n": 120, "mu": 0.0, "sigma": 0.3, "regime": 1},
    ]
    r, labels = regime_series(segs, seed=0)
    assert r.size == 220
    assert labels.size == 220
    assert set(labels) == {0, 1}


def test_regime_vol_differs():
    segs = [
        {"kind": "gbm", "n": 500, "mu": 0.1, "sigma": 0.05, "regime": 0},
        {"kind": "gbm", "n": 500, "mu": 0.0, "sigma": 0.30, "regime": 1},
    ]
    r, labels = regime_series(segs, seed=2)
    assert r[labels == 1].std() > r[labels == 0].std()


def test_demo_series_consistent_shapes():
    dates, prices, returns, labels = demo_series()
    assert len(dates) == len(prices) == len(returns) == len(labels)
    assert (prices > 0).all()


def test_log_returns_length():
    prices = np.array([100.0, 101.0, 102.0, 100.0])
    r = log_returns(prices)
    assert r.size == 3
