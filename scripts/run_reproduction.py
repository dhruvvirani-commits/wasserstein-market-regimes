import sys
import numpy as np

from src.data import load_prices_yf, log_returns
from src.regimes import fit_regimes, distance_matrix, silhouette
from src.wkmeans import MomentKMeans
from src.mmd import cluster_mmd


def main(ticker="SPY", start="2015-01-01", end="2024-12-31", k=4, w=21):
    prices = load_prices_yf(ticker, start, end)
    returns = log_returns(prices.values)
    model, windows, starts = fit_regimes(returns, w=w, step=w, k=k, random_state=0)
    dmat = distance_matrix(windows, p=2)

    wk = cluster_mmd(windows, model.labels_)
    mom = MomentKMeans(k=k, random_state=0).fit(windows)
    mom_mmd = cluster_mmd(windows, mom.labels_)
    sil = silhouette(dmat, model.labels_)

    print(f"ticker          {ticker}  ({start} -> {end})")
    print(f"windows         {len(windows)}")
    print(f"regimes (k)     {k}")
    print(f"silhouette      {sil:.3f}")
    print(f"WK-means MMD    between={wk['between']:.4f}  within={wk['within']:.4f}  ratio={wk['ratio']:.2f}")
    print(f"Moment MMD      between={mom_mmd['between']:.4f}  within={mom_mmd['within']:.4f}  ratio={mom_mmd['ratio']:.2f}")
    print(f"uplift          {wk['ratio'] / mom_mmd['ratio']:.2f}x")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()
