import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

from .wasserstein import quantile_functions, wasserstein_matrix
from .wkmeans import WassersteinKMeans


def rolling_windows(returns, w, step):
    r = np.asarray(returns, dtype=float)
    windows = []
    starts = []
    i = 0
    while i + w <= r.size:
        windows.append(r[i:i + w])
        starts.append(i)
        i += step
    return windows, np.array(starts)


def fit_regimes(returns, w=21, step=21, k=4, p=2, n_grid=256, random_state=0):
    windows, starts = rolling_windows(returns, w, step)
    model = WassersteinKMeans(k=k, p=p, n_grid=n_grid, random_state=random_state).fit(windows)
    return model, windows, starts


def labels_to_series(window_labels, starts, w, n):
    series = np.full(n, -1)
    for lab, s in zip(window_labels, starts):
        series[s:s + w] = lab
    if (series == -1).any():
        series[series == -1] = window_labels[-1]
    return series


def transition_matrix(window_labels, k):
    T = np.zeros((k, k))
    for a, b in zip(window_labels[:-1], window_labels[1:]):
        T[int(a), int(b)] += 1
    rows = T.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return T / rows


def distance_matrix(windows, p=2, n_grid=256):
    return wasserstein_matrix(quantile_functions(windows, n_grid), p)


def embed_2d(dmat, random_state=0):
    mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
              random_state=random_state, normalized_stress="auto")
    return mds.fit_transform(dmat)


def silhouette(dmat, labels):
    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(dmat, labels, metric="precomputed"))


def elbow_scan(windows, k_values, p=2, n_grid=256, random_state=0):
    inertias = []
    for k in k_values:
        m = WassersteinKMeans(k=k, p=p, n_grid=n_grid, random_state=random_state).fit(windows)
        inertias.append(m.inertia_)
    return list(k_values), inertias
