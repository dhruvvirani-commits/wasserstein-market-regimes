import sys
import numpy as np
import pandas as pd

from src.data import log_returns
from src.regimes import fit_regimes, labels_to_series
from src.mmd import cluster_mmd


def load_mt5_csv(path, price_col="Close"):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    col = cols.get(price_col.lower(), price_col)
    return df[col].astype(float).values


def main(price_csv, trades_csv=None, k=3, w=21):
    k = int(k)
    w = int(w)
    prices = load_mt5_csv(price_csv)
    returns = log_returns(prices)
    model, windows, starts = fit_regimes(returns, w=w, step=w, k=k, random_state=0)
    series = labels_to_series(model.labels_, starts, w, returns.size)

    wk = cluster_mmd(windows, model.labels_)
    print(f"windows         {len(windows)}")
    print(f"regimes (k)     {k}")
    print(f"WK-means ratio  {wk['ratio']:.2f}")
    for r in range(k):
        share = np.mean(series == r) * 100
        print(f"regime {r}        {share:5.1f}% of days")

    if trades_csv:
        trades = pd.read_csv(trades_csv)
        idx_col = [c for c in trades.columns if c.lower() in ("bar", "index", "i")]
        pnl_col = [c for c in trades.columns if c.lower() in ("r", "pnl", "return")]
        if idx_col and pnl_col:
            trades["regime"] = trades[idx_col[0]].clip(0, len(series) - 1).map(lambda i: series[int(i)])
            table = trades.groupby("regime")[pnl_col[0]].agg(["count", "mean", "sum"])
            print("\nStrategy expectancy by regime")
            print(table.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: python scripts/run_extension.py PRICE_CSV [TRADES_CSV] [k] [w]")
    else:
        main(*args)
