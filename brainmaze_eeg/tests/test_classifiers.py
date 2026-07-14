"""
Import + construction smoke tests for brainmaze_eeg.classifiers.

The module was left unimportable / unconstructible by the best -> brainmaze migration
(#40). These tests pin that it imports and that every public class constructs with
sensible defaults, so the migration cannot silently regress. They intentionally do NOT
exercise fit/predict -- several methods still have runtime bugs tracked separately.
"""
import warnings

import numpy as np
import pytest


def test_module_imports():
    import brainmaze_eeg.classifiers  # noqa: F401


import brainmaze_eeg.classifiers as C


MODEL_CLASSES = [
    C.KDEBayesianModel,
    C.KDEBayesianCausalModel,
    C.KDEBayesianModelNC,
    C.MVGaussBayesianModel,
    C.MVGaussBayesianCausalModel,
    C.MultiChannelMVGaussBayesClassifier,
]


@pytest.mark.parametrize('cls', MODEL_CLASSES, ids=lambda c: c.__name__)
def test_model_constructs_with_defaults(cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = cls(fs=200, segm_size=30)
    # the sub-extractors must have been built (this is where the kwarg drift crashed)
    assert model.FeatureExtractor is not None
    assert model.FeatureExtractor_MeanBand is not None


@pytest.mark.parametrize('cls', MODEL_CLASSES, ids=lambda c: c.__name__)
def test_bands_to_erase_flows_into_extractor_ignore_bands(cls):
    # `bands_to_erase` is the model-facing name; the extractor now calls it `ignore_bands`.
    # The migration must forward it, not drop it.
    erase = [[6, 8], [13, 15]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = cls(fs=200, segm_size=30, bands_to_erase=erase)
    assert model.FeatureExtractor._ignore_bands == erase
    assert model.FeatureExtractor_MeanBand._ignore_bands == erase


def test_dead_extractor_kwargs_are_not_forwarded():
    # filter_bands / filter_order are kept on the model for API compatibility but must
    # not be passed to the extractor (which no longer accepts them).
    import inspect
    from brainmaze_eeg.features.feature_extraction import SleepSpectralFeatureExtractor
    params = set(inspect.signature(SleepSpectralFeatureExtractor.__init__).parameters)
    assert 'bands_to_erase' not in params
    assert 'filter_bands' not in params
    assert 'nfiltorder' not in params
    # the model still exposes them
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = C.KDEBayesianModel(fs=200, segm_size=30)
    assert hasattr(model, 'filter_bands')


def test_non_model_helpers_construct():
    C.SleepStageProbabilityMarkovChainFilter()
    C.Mapper()
    C.SleepStructureClassifier()
    C.SleepClassifierWrapper()


def test_multivariate_normal_wrapper_fits_from_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 100))       # (n_features, n_samples) per the wrapper's convention
    dist = C.multivariate_normal_(X)
    assert np.isfinite(dist.pdf(X)).all()


def test_markov_filter_transition_matrix_rows_normalised():
    mf = C.SleepStageProbabilityMarkovChainFilter()
    np.testing.assert_allclose(mf.tmat.sum(axis=1), 1.0, atol=1e-9)
