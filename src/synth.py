import numpy as np


def gbm_returns(n, mu, sigma, dt=1.0 / 252.0, seed=0):
    rng = np.random.default_rng(seed)
    return (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)


def merton_returns(n, mu, sigma, lam, jump_mu, jump_sigma, dt=1.0 / 252.0, seed=0):
    rng = np.random.default_rng(seed)
    diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    counts = rng.poisson(lam * dt, n)
    jumps = np.array([rng.normal(jump_mu, jump_sigma, c).sum() if c > 0 else 0.0 for c in counts])
    return diffusion + jumps


def regime_series(segments, seed=0):
    returns = []
    labels = []
    for i, seg in enumerate(segments):
        s = seed + i * 17
        if seg["kind"] == "gbm":
            r = gbm_returns(seg["n"], seg["mu"], seg["sigma"], seed=s)
        else:
            r = merton_returns(seg["n"], seg["mu"], seg["sigma"], seg["lam"],
                               seg["jump_mu"], seg["jump_sigma"], seed=s)
        returns.append(r)
        labels.append(np.full(seg["n"], seg.get("regime", i)))
    return np.concatenate(returns), np.concatenate(labels)
