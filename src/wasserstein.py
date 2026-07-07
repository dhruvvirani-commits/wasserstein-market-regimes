import numpy as np


def quantile_grid(n_grid):
    return (np.arange(n_grid) + 0.5) / n_grid


def _sorted_quantiles(samples, grid):
    s = np.sort(np.asarray(samples, dtype=float))
    if s.size == 0:
        return np.zeros_like(grid)
    q = (np.arange(s.size) + 0.5) / s.size
    return np.interp(grid, q, s)


def quantile_functions(sample_list, n_grid=256):
    grid = quantile_grid(n_grid)
    return np.vstack([_sorted_quantiles(s, grid) for s in sample_list])


def p_wasserstein(x, y, p=2, n_grid=256):
    grid = quantile_grid(n_grid)
    fx = _sorted_quantiles(x, grid)
    fy = _sorted_quantiles(y, grid)
    return float(np.mean(np.abs(fx - fy) ** p) ** (1.0 / p))


def wasserstein_matrix(quantiles, p=2):
    n = quantiles.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.mean(np.abs(quantiles[i] - quantiles) ** p, axis=1) ** (1.0 / p)
    np.fill_diagonal(d, 0.0)
    return d


def wasserstein_to_centroids(quantiles, centroids, p=2):
    n = quantiles.shape[0]
    k = centroids.shape[0]
    d = np.zeros((n, k))
    for j in range(k):
        d[:, j] = np.mean(np.abs(quantiles - centroids[j]) ** p, axis=1) ** (1.0 / p)
    return d
