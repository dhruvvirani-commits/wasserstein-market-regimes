

# P04 - Market Regime Clustering with the Wasserstein Distance

Unsupervised detection of market regimes by clustering **entire empirical return distributions** using optimal transport, rather than a handful of scalar features. This is a faithful reproduction of Horvath, Issa and Muguruza (2021), extended onto an FX book and wired into an interactive dashboard.

## Why this exists

Standard regime detection compresses each slice of market history into mean and volatility and clusters those numbers. That discards the shape of the distribution: skew, fat tails, bimodality, the very things that separate a calm market from a fragile one. Here the whole empirical return distribution is the object of study, and the distance between two market states is the p-Wasserstein distance from optimal transport. Clustering then happens directly on the space of probability measures.

## Method

1. Slide a window across the log-return series. Each window becomes an empirical distribution.
2. Compute the p-Wasserstein distance between windows. In one dimension this has a closed form through the quantile functions, so it is exact and cheap.
3. Run Wasserstein k-means: assign each window to its nearest centroid, recompute each centroid as the Wasserstein barycenter of its members, repeat to convergence.
4. Validate separation with the maximum mean discrepancy (MMD) between and within clusters, benchmarked against a moment-based k-means.

## Results on the bundled demo series

A synthetic series with three planted regimes (calm, choppy, stress) recovers cleanly:

| Metric | WK-means | Moment k-means |
|---|---|---|
| MMD separation ratio | 28.3x | 4.7x |
| Silhouette | 0.44 | - |

WK-means separates the regimes roughly 6x more sharply than the moment baseline, matching the paper's central claim.

## Layout

```
src/
  wasserstein.py    p-Wasserstein distance, quantile-based
  wkmeans.py        Wasserstein k-means, barycenters, moment benchmark
  mmd.py            maximum mean discrepancy scoring
  synth.py          GBM and Merton jump-diffusion generators
  data.py           yfinance loader, log returns, demo series
  regimes.py        windowing, mapping, transitions, validation
scripts/
  build_dashboard.py       builds the interactive HTML dashboard
  build_dashboard_pro.py   builds the pro terminal (live controls + inspector)
  run_reproduction.py  WK-means vs moment baseline on SPY
  run_extension.py     regimes on EUR/USD, expectancy by regime
tests/                 29 tests
```

## Run it

```bash
pip install -r requirements.txt

python -m pytest -q

python -m scripts.build_dashboard_pro
python -m scripts.build_dashboard_pro SPY 2015-01-01 2024-12-31
python -m scripts.build_dashboard_pro csv eurusd.csv
python -m scripts.run_reproduction SPY 2015-01-01 2024-12-31 4 21
python -m scripts.run_extension eurusd.csv trades.csv 3 21
```

The dashboard writes a self-contained HTML file with seven interactive panels: a regime-shaded price timeline, per-cluster return ridgelines, the Wasserstein distance matrix, a 2D MDS regime map, the transition matrix, the WK-means versus baseline comparison, and k-selection by inertia and silhouette.

## Reference

Horvath, B., Issa, Z., and Muguruza, A. (2021). Clustering Market Regimes using the Wasserstein Distance. arXiv:2110.11848. Published in the Journal of Computational Finance 28(1), 1-39 (2024).

This implementation is for research and educational purposes. Not financial advice.
