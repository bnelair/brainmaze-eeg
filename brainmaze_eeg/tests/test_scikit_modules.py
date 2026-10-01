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


# --------------------------------------------------------------------------- PCAModuleSVD (#39)
from brainmaze_eeg.scikit_modules import PCAModuleSVD


def _eig_unordered_cov_data():
    """Centred 8-feature data whose covariance makes ``np.linalg.eig`` return eigenvalues
    out of order ([.., 0.896, 0.292, 0.589, 0.449] on numpy 1.26 and 2.x) -- the case the
    pre-#39 ``eig``-based ``PCAModuleSVD`` silently mis-handled."""
    r = np.random.default_rng(0)
    d = int(r.integers(3, 9))
    Z = r.normal(size=(200, d)) * r.uniform(0.1, 5, size=d)
    return Z - Z.mean(axis=0)


def test_pcamodulesvd_eigenvalues_real_sorted_descending():
    X = _eig_unordered_cov_data()
    m = PCAModuleSVD(var_threshold=0.98).fit(X)
    assert np.isrealobj(m.eigen_vals) and np.isrealobj(m.eigen_vecs)
    assert np.all(np.diff(m.eigen_vals) <= 0)
    assert np.all(m.eigen_vals >= 0)


def test_pcamodulesvd_matches_sklearn_pca_on_centred_data():
    X = _eig_unordered_cov_data()
    m = PCAModuleSVD(var_threshold=0.98).fit(X)
    ref = PCA().fit(X)
    np.testing.assert_allclose(m.eigen_vals, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(m.explained_variance_ratio, ref.explained_variance_ratio_, rtol=1e-8)
    # same component directions (up to sign)
    np.testing.assert_allclose(np.abs(m.eigen_vecs.T @ ref.components_.T), np.eye(X.shape[1]), atol=1e-6)
    # same component count as the explained-variance rule in PCAModule
    assert m.n == PCAModule(var_threshold=0.98).fit(X).n_components_
    # transform = projection on the leading components (up to sign)
    np.testing.assert_allclose(np.abs(m.transform(X)), np.abs(ref.transform(X)[:, :m.n]), atol=1e-8)


def test_pcamodulesvd_threshold_semantics():
    # 2 components explain 99.02 %: the fewest components reaching the threshold
    X = _data([10, 5, 1, 0.3, 0.1, 0.05]); X = X - X.mean(axis=0)
    assert PCAModuleSVD(0.98).fit(X).n == 2
    assert PCAModuleSVD(0.5).fit(X).n == 1
    # threshold 1.0 terminates (old while-loop could spin forever on round-off) and keeps all
    assert PCAModuleSVD(1.0).fit(X).n == X.shape[1]
    assert PCAModuleSVD(0.98).fit_transform(X).shape == (X.shape[0], 2)


def test_pca_svd_sign_convention():
    # review R7 of PR #69: deterministic eigenvector signs (like sklearn's svd_flip)
    from brainmaze_eeg.scikit_modules import PCAModuleSVD
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))
    X -= X.mean(axis=0)
    p = PCAModuleSVD(var_threshold=1.0).fit(X)
    V = p.eigen_vecs
    piv = np.argmax(np.abs(V), axis=0)
    assert np.all(V[piv, np.arange(V.shape[1])] > 0)
    # same components as sklearn up to that sign convention
    from sklearn.decomposition import PCA
    comp = PCA().fit(X).components_.T
    comp = comp * np.sign(comp[np.argmax(np.abs(comp), axis=0), np.arange(comp.shape[1])])
    np.testing.assert_allclose(V, comp, atol=1e-8)
