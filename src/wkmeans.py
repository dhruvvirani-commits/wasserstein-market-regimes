import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans

from .wasserstein import quantile_functions, wasserstein_to_centroids


class WassersteinKMeans:
    def __init__(self, k=4, p=2, n_grid=256, max_iter=100, n_init=8, random_state=0):
        self.k = k
        self.p = p
        self.n_grid = n_grid
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _init_centroids(self, Q, rng):
        n = Q.shape[0]
        idx = [int(rng.integers(n))]
        closest = np.full(n, np.inf)
        for _ in range(1, self.k):
            last = Q[idx[-1]]
            dist = np.mean(np.abs(Q - last) ** self.p, axis=1) ** (1.0 / self.p)
            closest = np.minimum(closest, dist ** 2)
            total = closest.sum()
            if total <= 0:
                idx.append(int(rng.integers(n)))
            else:
                idx.append(int(rng.choice(n, p=closest / total)))
        return Q[idx].copy()

    def _run_single(self, Q, rng):
        C = self._init_centroids(Q, rng)
        labels = np.full(Q.shape[0], -1)
        for _ in range(self.max_iter):
            D = wasserstein_to_centroids(Q, C, self.p)
            new_labels = D.argmin(axis=1)
            newC = C.copy()
            for j in range(self.k):
                members = Q[new_labels == j]
                if members.shape[0] > 0:
                    newC[j] = members.mean(axis=0)
            shift = np.abs(newC - C).max()
            C = newC
            if np.array_equal(new_labels, labels) and shift < 1e-10:
                labels = new_labels
                break
            labels = new_labels
        inertia = float(wasserstein_to_centroids(Q, C, self.p).min(axis=1).sum())
        return labels, C, inertia

    def fit(self, sample_list):
        Q = quantile_functions(sample_list, self.n_grid)
        best = None
        for s in range(self.n_init):
            rng = np.random.default_rng(self.random_state + s)
            labels, C, inertia = self._run_single(Q, rng)
            if best is None or inertia < best[2]:
                best = (labels, C, inertia)
        self.labels_ = best[0]
        self.centroids_ = best[1]
        self.inertia_ = best[2]
        self.quantiles_ = Q
        self._order_by_dispersion()
        return self

    def _order_by_dispersion(self):
        spread = self.centroids_[:, -1] - self.centroids_[:, 0]
        order = np.argsort(spread)
        remap = np.zeros(self.k, dtype=int)
        for new, old in enumerate(order):
            remap[old] = new
        self.centroids_ = self.centroids_[order]
        self.labels_ = remap[self.labels_]

    def predict(self, sample_list):
        Q = quantile_functions(sample_list, self.n_grid)
        return wasserstein_to_centroids(Q, self.centroids_, self.p).argmin(axis=1)


def moment_features(sample_list):
    feats = []
    for s in sample_list:
        s = np.asarray(s, dtype=float)
        feats.append([s.mean(), s.std(), float(skew(s)), float(kurtosis(s))])
    return np.array(feats)


class MomentKMeans:
    def __init__(self, k=4, random_state=0):
        self.k = k
        self.random_state = random_state

    def fit(self, sample_list):
        X = moment_features(sample_list)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        km = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state).fit(Xs)
        self.labels_ = km.labels_
        self.features_ = X
        return self
