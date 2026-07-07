import numpy as np


def median_gamma(values, cap=2000, seed=0):
    v = np.asarray(values, dtype=float)
    if v.size > cap:
        idx = np.random.default_rng(seed).choice(v.size, cap, replace=False)
        v = v[idx]
    d = np.abs(v[:, None] - v[None, :])
    positive = d[d > 0]
    med = np.median(positive) if positive.size else 1.0
    if med == 0:
        med = 1.0
    return 1.0 / (2.0 * med ** 2)


def _kernel_mean(a, b, gamma):
    d2 = (a[:, None] - b[None, :]) ** 2
    return float(np.exp(-gamma * d2).mean())


def mmd2(x, y, gamma):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return _kernel_mean(x, x, gamma) + _kernel_mean(y, y, gamma) - 2.0 * _kernel_mean(x, y, gamma)


def cluster_mmd(sample_list, labels, gamma=None, seed=0):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    pooled = np.concatenate([np.asarray(s, dtype=float) for s in sample_list])
    if gamma is None:
        gamma = median_gamma(pooled, seed=seed)
    groups = {}
    for u in uniq:
        members = [np.asarray(sample_list[i], dtype=float)
                   for i in range(len(sample_list)) if labels[i] == u]
        groups[u] = np.concatenate(members)
    within = []
    for u in uniq:
        g = groups[u]
        if g.size > 4:
            half = g.size // 2
            within.append(mmd2(g[:half], g[half:], gamma))
    between = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            between.append(mmd2(groups[uniq[i]], groups[uniq[j]], gamma))
    w = float(np.mean(within)) if within else 0.0
    b = float(np.mean(between)) if between else 0.0
    ratio = b / w if w > 1e-12 else float("inf")
    return {"within": w, "between": b, "ratio": ratio, "gamma": float(gamma)}
