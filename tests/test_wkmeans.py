import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.wkmeans import WassersteinKMeans, MomentKMeans, moment_features


def _two_group_windows(seed=0):
    rng = np.random.default_rng(seed)
    calm = [rng.normal(0, 0.01, 60) for _ in range(15)]
    wild = [rng.normal(0, 0.05, 60) for _ in range(15)]
    windows = calm + wild
    truth = np.array([0] * 15 + [1] * 15)
    return windows, truth


def test_recovers_two_groups():
    windows, truth = _two_group_windows()
    model = WassersteinKMeans(k=2, p=2, random_state=0).fit(windows)
    assert adjusted_rand_score(truth, model.labels_) > 0.9


def test_label_count_matches_windows():
    windows, _ = _two_group_windows(1)
    model = WassersteinKMeans(k=2, random_state=1).fit(windows)
    assert model.labels_.shape[0] == len(windows)
    assert set(model.labels_).issubset({0, 1})


def test_centroids_ordered_by_dispersion():
    windows, _ = _two_group_windows(2)
    model = WassersteinKMeans(k=2, random_state=2).fit(windows)
    spread = model.centroids_[:, -1] - model.centroids_[:, 0]
    assert spread[0] <= spread[1]


def test_predict_assigns_new_windows():
    windows, _ = _two_group_windows(3)
    model = WassersteinKMeans(k=2, random_state=3).fit(windows)
    rng = np.random.default_rng(9)
    new = [rng.normal(0, 0.01, 60), rng.normal(0, 0.05, 60)]
    preds = model.predict(new)
    assert preds[0] != preds[1]


def test_moment_features_shape():
    windows, _ = _two_group_windows(4)
    feats = moment_features(windows)
    assert feats.shape == (len(windows), 4)


def test_moment_kmeans_runs():
    windows, truth = _two_group_windows(5)
    model = MomentKMeans(k=2, random_state=0).fit(windows)
    assert model.labels_.shape[0] == len(windows)
