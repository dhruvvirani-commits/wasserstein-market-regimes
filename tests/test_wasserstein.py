import numpy as np

from src.wasserstein import (
    p_wasserstein,
    wasserstein_matrix,
    quantile_functions,
    wasserstein_to_centroids,
)


def test_identical_distributions_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert p_wasserstein(x, x, p=2) == 0.0


def test_point_masses_equal_shift():
    a = np.full(100, 3.0)
    b = np.full(100, 5.0)
    assert abs(p_wasserstein(a, b, p=1) - 2.0) < 1e-6
    assert abs(p_wasserstein(a, b, p=2) - 2.0) < 1e-6


def test_symmetry():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    y = rng.normal(1, 2, 300)
    assert abs(p_wasserstein(x, y) - p_wasserstein(y, x)) < 1e-9


def test_translation_invariance_of_shape():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    shift = 4.0
    d = p_wasserstein(x, x + shift, p=1)
    assert abs(d - shift) < 0.05


def test_matrix_shape_and_diagonal():
    rng = np.random.default_rng(3)
    samples = [rng.normal(m, 1, 200) for m in range(5)]
    Q = quantile_functions(samples, n_grid=128)
    D = wasserstein_matrix(Q, p=2)
    assert D.shape == (5, 5)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)


def test_to_centroids_matches_pairwise():
    rng = np.random.default_rng(4)
    samples = [rng.normal(0, 1, 200), rng.normal(3, 1, 200)]
    Q = quantile_functions(samples, n_grid=128)
    D = wasserstein_to_centroids(Q, Q, p=2)
    assert D.shape == (2, 2)
    assert D[0, 0] < D[0, 1]
