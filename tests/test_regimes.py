import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.regimes import (
    rolling_windows,
    fit_regimes,
    labels_to_series,
    transition_matrix,
    distance_matrix,
    silhouette,
    elbow_scan,
)
from src.synth import regime_series


def test_rolling_windows_count():
    r = np.arange(100.0)
    windows, starts = rolling_windows(r, w=20, step=20)
    assert len(windows) == 5
    assert starts.tolist() == [0, 20, 40, 60, 80]


def test_labels_to_series_full_coverage():
    window_labels = np.array([0, 1, 2])
    starts = np.array([0, 20, 40])
    series = labels_to_series(window_labels, starts, w=20, n=60)
    assert (series != -1).all()
    assert series[0] == 0 and series[25] == 1 and series[50] == 2


def test_transition_matrix_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1, 2, 0, 1])
    T = transition_matrix(labels, k=3)
    assert np.allclose(T.sum(axis=1), 1.0)


def test_distance_matrix_symmetric():
    rng = np.random.default_rng(0)
    windows = [rng.normal(0, s, 60) for s in [0.01, 0.02, 0.05, 0.08]]
    D = distance_matrix(windows, p=2, n_grid=128)
    assert np.allclose(D, D.T)


def test_pipeline_recovers_planted_regimes():
    segs = [
        {"kind": "gbm", "n": 315, "mu": 0.12, "sigma": 0.08, "regime": 0},
        {"kind": "gbm", "n": 315, "mu": 0.00, "sigma": 0.30, "regime": 1},
        {"kind": "gbm", "n": 315, "mu": 0.12, "sigma": 0.08, "regime": 0},
        {"kind": "gbm", "n": 315, "mu": 0.00, "sigma": 0.30, "regime": 1},
    ]
    returns, truth = regime_series(segs, seed=0)
    model, windows, starts = fit_regimes(returns, w=21, step=21, k=2, random_state=0)
    series = labels_to_series(model.labels_, starts, 21, returns.size)
    valid = series != -1
    assert adjusted_rand_score(truth[valid], series[valid]) > 0.7


def test_silhouette_in_range():
    rng = np.random.default_rng(1)
    windows = [rng.normal(0, 0.01, 60) for _ in range(10)] + \
              [rng.normal(0, 0.05, 60) for _ in range(10)]
    labels = np.array([0] * 10 + [1] * 10)
    D = distance_matrix(windows, n_grid=128)
    s = silhouette(D, labels)
    assert -1.0 <= s <= 1.0


def test_elbow_scan_returns_matched_lengths():
    rng = np.random.default_rng(2)
    windows = [rng.normal(0, s, 60) for s in rng.uniform(0.01, 0.06, 20)]
    ks, inertias = elbow_scan(windows, [2, 3, 4], n_grid=128)
    assert len(ks) == len(inertias) == 3
    assert inertias[0] >= inertias[-1]
