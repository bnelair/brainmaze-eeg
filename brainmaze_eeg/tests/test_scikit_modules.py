import numpy as np
import pytest
from sklearn.decomposition import PCA

from brainmaze_eeg.scikit_modules import PCAModule


def _data(scales, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, len(scales))) @ np.diag(scales)


def test_pcamodule_selects_components_by_explained_variance():
    # 2 components explain 99.02% here; the threshold must select 2, not 4.
    X = _data([10, 5, 1, 0.3, 0.1, 0.05])
    evr = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    true_n = int(np.searchsorted(evr, 0.98) + 1)

    model = PCAModule(var_threshold=0.98)
    model.fit(X)
    assert model.n_components_ == true_n == 2
    assert model.transform(X).shape == (X.shape[0], true_n)


def test_pcamodule_higher_threshold_keeps_more_components():
    X = _data([10, 5, 1, 0.3, 0.1, 0.05])
    n_low = PCAModule(var_threshold=0.90).fit(X).n_components_
    n_high = PCAModule(var_threshold=0.999).fit(X).n_components_
    assert n_high > n_low


def test_pcamodule_threshold_one_keeps_all_components():
    X = _data([5, 4, 3, 2, 1])
    model = PCAModule(var_threshold=1.0).fit(X)
    assert model.n_components_ == X.shape[1]


def test_pcamodule_fit_transform_matches_fit_then_transform():
    X = _data([8, 4, 2, 1, 0.5])
    a = PCAModule(var_threshold=0.95).fit_transform(X)
    m = PCAModule(var_threshold=0.95); m.fit(X)
    np.testing.assert_allclose(a, m.transform(X))
