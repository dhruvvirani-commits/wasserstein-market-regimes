import numpy as np

from src.mmd import mmd2, cluster_mmd, median_gamma


def test_mmd_same_distribution_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    g = median_gamma(np.concatenate([x, y]))
    assert mmd2(x, y, g) < 0.02


def test_mmd_different_distributions_larger():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    y = rng.normal(5, 1, 500)
    g = median_gamma(np.concatenate([x, y]))
    same = mmd2(x, rng.normal(0, 1, 500), g)
    diff = mmd2(x, y, g)
    assert diff > same


def test_median_gamma_positive():
    rng = np.random.default_rng(2)
    assert median_gamma(rng.normal(0, 1, 300)) > 0


def test_cluster_mmd_separation():
    rng = np.random.default_rng(3)
    calm = [rng.normal(0, 0.01, 80) for _ in range(10)]
    wild = [rng.normal(0, 0.05, 80) for _ in range(10)]
    samples = calm + wild
    labels = np.array([0] * 10 + [1] * 10)
    result = cluster_mmd(samples, labels)
    assert result["between"] > result["within"]
    assert result["ratio"] > 1.0
