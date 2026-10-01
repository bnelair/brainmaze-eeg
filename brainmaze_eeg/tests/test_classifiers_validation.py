"""
End-to-end validation of every classifier in ``brainmaze_eeg.classifiers`` (issue #57).

Each model is fitted on small synthetic data with known, separable class structure and
checked for: output shapes and label sets, accuracy well above chance on held-out data,
determinism (refitting gives identical output), and no train/test leakage in the
normalisation. Raw-signal tests run the full ``extract_features_bulk -> fit ->
predict_signal`` path.

Synthetic data
--------------
* Feature level: ``d`` Gaussian features (unit variance); state ``i`` of
  ``AWAKE, N2, N3, REM`` is shifted by ``SEP`` (3 sd) along feature ``i``. Labels follow a
  physiologically ordered hypnogram (AWAKE -> N2 -> N3 -> N2 -> REM -> ...), blocks of
  6-11 epochs, because the KDE models smooth scores across consecutive epochs and the
  causal models apply a sleep-stage transition prior (which strongly penalises e.g.
  AWAKE -> N3).
* Raw level: per epoch, a dominant sinusoid (AWAKE 10 Hz, N2 13 Hz, N3 1.5 Hz high
  amplitude, REM 6 Hz) + a weaker second one + white noise.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import brainmaze_eeg.classifiers as C
from brainmaze_eeg.features.utils import balance_classes

STATES = ['AWAKE', 'N2', 'N3', 'REM']
CYCLE = ['AWAKE', 'N2', 'N3', 'N2', 'REM']
SEP = 3.0
CHANCE = 1.0 / len(STATES)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def cyc_hypnogram(n_blocks, seed, lo=6, hi=12):
    r = np.random.default_rng(seed)
    return np.concatenate([[CYCLE[k % len(CYCLE)]] * int(r.integers(lo, hi)) for k in range(n_blocks)])


def features(y, d=10, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(len(y), d))
    for i, s in enumerate(STATES[:d]):
        X[np.asarray(y) == s, i] += SEP
    return X


@pytest.fixture(scope='module')
def feature_data():
    ytr = cyc_hypnogram(20, 0)          # ~170 epochs, unbalanced (N2 twice as frequent)
    yte = cyc_hypnogram(15, 1)          # held-out recording, different noise
    return features(ytr, seed=0), ytr, features(yte, seed=1), yte


KDE_FAMILY = [
    C.KDEBayesianModel,
    C.KDEBayesianModelNC,
    C.MVGaussBayesianModel,
    C.KDEBayesianCausalModel,
    C.MVGaussBayesianCausalModel,
]
ALL_MODELS = KDE_FAMILY + [C.MultiChannelMVGaussBayesClassifier]


def make(cls, **kw):
    if cls is not C.MultiChannelMVGaussBayesClassifier:
        kw.setdefault('n_jobs', 1)
    return cls(**kw)


# --------------------------------------------------------------------------- feature-level models
@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_model_fit_predict_valid(cls, feature_data):
    Xtr, ytr, Xte, yte = feature_data
    m = make(cls)
    m.fit(Xtr, ytr)

    pred = m.predict(Xte)
    assert pred.shape == (Xte.shape[0],)
    assert set(pred) <= set(STATES)

    sc = m.scores(Xte)
    assert isinstance(sc, pd.DataFrame)
    assert sc.shape == (Xte.shape[0], len(STATES))
    assert sorted(sc.columns) == STATES
    assert np.all(np.isfinite(sc.to_numpy(dtype=float)))
    np.testing.assert_allclose(sc.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(pred, np.asarray(sc.idxmax(axis=1)))

    acc = np.mean(pred == yte)
    assert acc > 0.85, f'{cls.__name__}: held-out accuracy {acc:.3f} (chance {CHANCE})'


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_model_is_deterministic(cls, feature_data):
    Xtr, ytr, Xte, _ = feature_data
    a, b = make(cls), make(cls)
    a.fit(Xtr, ytr)
    b.fit(Xtr, ytr)
    np.testing.assert_array_equal(a.predict(Xte), b.predict(Xte))
    np.testing.assert_array_equal(a.scores(Xte).to_numpy(), b.scores(Xte).to_numpy())


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_model_name_matches_class(cls):
    # up to v1.0.0 several classes reported a copy-pasted ``__name__``
    assert make(cls).__name__ == cls.__name__


def test_zscore_fitted_on_balanced_training_data_and_frozen(feature_data):
    """No leakage and no silent refit: the z-score statistics are those of the balanced
    training data (in the selected/PCA space) and are not touched by prediction."""
    Xtr, ytr, Xte, _ = feature_data
    m = make(C.KDEBayesianModel)
    m.fit(Xtr, ytr)

    Xb, _ = balance_classes(Xtr.copy(), ytr.copy(), std_factor=0.0)
    Zb = m.PCA.transform(m.SELECTOR.transform(Xb))
    np.testing.assert_allclose(m.ZScore.mean, Zb.mean(axis=0, keepdims=True), atol=1e-10)
    np.testing.assert_allclose(m.ZScore.std, Zb.std(axis=0, keepdims=True), rtol=1e-10)

    mean0, std0 = m.ZScore.mean.copy(), m.ZScore.std.copy()
    comp0 = m.PCA.components_.copy()
    m.predict(Xte + 50.0)           # wildly shifted test data
    np.testing.assert_array_equal(m.ZScore.mean, mean0)
    np.testing.assert_array_equal(m.ZScore.std, std0)
    np.testing.assert_array_equal(m.PCA.components_, comp0)


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.MVGaussBayesianModel,
                                 C.MultiChannelMVGaussBayesClassifier], ids=lambda c: c.__name__)
def test_rows_scored_independently_without_smoothing(cls, feature_data):
    """With smoothing disabled a test row's score must not depend on the other test rows
    (no test-batch statistics anywhere in the pipeline)."""
    Xtr, ytr, Xte, _ = feature_data
    m = make(cls, window_smooth_n=1)
    m.fit(Xtr, ytr)
    batch = m.scores(Xte).to_numpy()
    single = np.vstack([m.scores(Xte[k:k + 1]).to_numpy() for k in range(0, Xte.shape[0], 7)])
    np.testing.assert_allclose(single, batch[::7], rtol=1e-9, atol=1e-12)


def test_scores_survive_density_underflow(feature_data):
    # a sample far from all training data: every class pdf underflows to 0. Up to v1.0.0
    # this gave 0/0 = NaN and ``predict`` raised; scores are now computed in log space.
    Xtr, ytr, Xte, _ = feature_data
    m = make(C.KDEBayesianModel)
    m.fit(Xtr, ytr)
    X = Xte.copy()
    X[20] += 1e3
    sc = m.scores(X)
    assert np.all(np.isfinite(sc.to_numpy(dtype=float)))
    assert m.predict(X).shape == (X.shape[0],)


def test_scores_on_few_epochs(feature_data):
    # filtfilt needed > 9 rows with the default window; up to v1.0.0 scoring <= 9 epochs raised
    Xtr, ytr, Xte, _ = feature_data
    m = make(C.KDEBayesianModel)
    m.fit(Xtr, ytr)
    for n in (1, 2, 5, 9):
        sc = m.scores(Xte[:n])
        assert sc.shape == (n, len(STATES))
        np.testing.assert_allclose(sc.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize('cls', [C.KDEBayesianCausalModel, C.MVGaussBayesianCausalModel],
                         ids=lambda c: c.__name__)
def test_causal_model_without_awake(cls, feature_data):
    # up to v1.0.0 the Markov chain always started in 'AWAKE' -> IndexError if not trained
    Xtr, ytr, Xte, yte = feature_data
    keep_tr, keep_te = ytr != 'AWAKE', yte != 'AWAKE'
    m = make(cls)
    m.fit(Xtr[keep_tr], ytr[keep_tr])
    pred = m.predict(Xte[keep_te])
    assert set(pred) <= {'N2', 'N3', 'REM'}
    assert np.mean(pred == yte[keep_te]) > 0.85


def test_markov_filter_rejects_unknown_labels():
    mf = C.SleepStageProbabilityMarkovChainFilter()
    sc = pd.DataFrame({'WAKE': [0.9, 0.1], 'N2': [0.1, 0.9]})
    with pytest.raises(ValueError, match='WAKE'):
        mf.fit(sc, np.array(['WAKE', 'N2']))


def test_markov_filter_stability_finite_for_weak_scores():
    # a useless score: the ROC optimum is the +inf threshold -> stability was -inf -> NaN tmat
    mf = C.SleepStageProbabilityMarkovChainFilter()
    y = np.array(['N2', 'N3'] * 20)
    sc = pd.DataFrame({'N2': np.full(40, 0.5), 'N3': np.full(40, 0.5)})
    mf.fit(sc, y)
    assert np.all(np.isfinite(mf.stability))
    assert np.all(np.isfinite(mf.tmat))


def test_multichannel_fit_transform_and_scores_layout(feature_data):
    Xtr, ytr, _, _ = feature_data
    m = C.MultiChannelMVGaussBayesClassifier()
    out = m.fit_transform(Xtr, ytr)
    np.testing.assert_array_equal(out, Xtr)
    sc = m.scores(Xtr)
    assert sc.shape == (Xtr.shape[0], len(STATES))
    assert list(sc.columns) == list(m.classifier.classes_)


def test_multichannel_rejects_other_sampling_rate():
    m = C.MultiChannelMVGaussBayesClassifier(fs=100, segm_size=10)
    with pytest.raises(ValueError, match='fs=100'):
        m.extract_features_bulk([np.zeros(2000)], [200])


# --------------------------------------------------------------------------- SleepStructureClassifier
def test_sleep_structure_classifier_valid():
    y = np.repeat(STATES, 12)
    yt = np.repeat(STATES, 8)
    X, Xt = features(y, d=4, seed=2), features(yt, d=4, seed=3)
    m = C.SleepStructureClassifier(states=STATES)
    m.fit(X, y)
    sc = m.scores(Xt)
    assert sc.shape == (Xt.shape[0], len(STATES))
    assert list(sc.columns) == STATES
    np.testing.assert_allclose(sc.sum(axis=1), 1.0, atol=1e-9)
    pred = m.predict(Xt)
    assert pred.shape == (Xt.shape[0],)
    assert np.mean(pred == yt) > 0.8
    # deterministic
    m2 = C.SleepStructureClassifier(states=STATES).fit(X, y)
    np.testing.assert_array_equal(m2.scores(Xt).to_numpy(), sc.to_numpy())


def test_sleep_structure_classifier_drops_untrained_states():
    y = np.repeat(['N2', 'N3'], 6)
    X = features(y, d=3, seed=0)
    with pytest.warns(UserWarning, match='WAKE'):
        m = C.SleepStructureClassifier().fit(X, y)    # default states include WAKE/N1/REM
    assert m.STATES == ['N2', 'N3']
    assert set(m.predict(X)) <= {'N2', 'N3'}


# --------------------------------------------------------------------------- Mapper
def test_mapper_map_recovers_affine_shift():
    y = np.repeat(STATES, 15)
    x = features(y, d=4, seed=4)[:, :3]
    shifted = (x - x.mean(0)) / 1.1 + x.mean(0) + 2.0      # needs scale 1.1, translate -2
    mp = C.Mapper()
    mp.create_template(x)
    out = mp.map(shifted, 'rec1', seed=0, popsize=6, maxiter=25)
    assert out.shape == shifted.shape
    assert 'rec1' in mp.MAPS
    assert mp.MAPS['rec1']['cost'] < 0.1 * mp._cost(shifted)
    np.testing.assert_allclose(out.mean(0), x.mean(0), atol=0.5)
    # cached: second call reuses the stored map
    tr = mp.MAPS['rec1']['transformation']
    np.testing.assert_array_equal(mp.map(shifted, 'rec1'), mp._transform(shifted, tr))
    # deterministic with a seed
    mp2 = C.Mapper(); mp2.create_template(x)
    mp2.map(shifted, 'rec1', seed=0, popsize=6, maxiter=25)
    np.testing.assert_array_equal(mp2.MAPS['rec1']['transformation']['translate'], tr['translate'])


@pytest.mark.parametrize('ndim', [2, 4])
def test_mapper_fit_genetic_any_dimension(ndim):
    # parameter vector split was hard-coded for 3 dims up to v1.0.0
    x = np.random.default_rng(0).normal(size=(60, ndim))
    mp = C.Mapper(); mp.create_template(x)
    tr, cost = mp.fit_genetic(x + 1.0, seed=0, popsize=4, maxiter=5)
    assert tr['translate'].shape == (ndim,) and tr['scale'].shape == (ndim,)
    assert np.isfinite(cost)


def test_mapper_fit_map_uses_labels(monkeypatch):
    # up to v1.0.0 fit_map did ``y = deepcopy(x)``: the labels were replaced by features
    y = np.array(['N2'] * 30 + ['N3'] * 10)
    x = features(np.where(y == 'N2', 'N2', 'N3'), d=4, seed=5)[:, 1:3]
    mp = C.Mapper(); mp.N = 3
    mp.create_template(x, y)
    seen = []
    orig = mp._cost_semi_supervised
    monkeypatch.setattr(mp, '_cost_semi_supervised',
                        lambda x_, y_, bias={}: seen.append(np.asarray(y_).copy()) or orig(x_, y_, bias=bias))
    costs, transforms = mp.fit_map(x, y)
    assert costs.shape == (3,) and len(transforms) == 3 and np.all(np.isfinite(costs))
    assert seen and seen[0].ndim == 1 and set(seen[0]) == {'N2', 'N3'}
    assert (seen[0] == 'N2').sum() == (seen[0] == 'N3').sum() == 30     # balanced


# --------------------------------------------------------------------------- raw-signal pipelines
_FREQ = {'AWAKE': (10.0, 20.0, 2.0), 'N2': (13.0, 3.0, 1.5), 'N3': (1.5, 1.0, 6.0), 'REM': (6.0, 25.0, 2.0)}


def _epoch(state, fs, dur, r):
    t = np.arange(int(round(fs * dur))) / fs
    f1, f2, a1 = _FREQ[state]
    return (a1 * np.sin(2 * np.pi * f1 * t + r.uniform(0, 2 * np.pi))
            + 0.7 * np.sin(2 * np.pi * f2 * t + r.uniform(0, 2 * np.pi))
            + r.normal(size=t.size))


def _recording(labels, fs, dur, seed):
    r = np.random.default_rng(seed)
    return np.concatenate([_epoch(s, fs, dur, r) for s in labels])


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.MultiChannelMVGaussBayesClassifier],
                         ids=lambda c: c.__name__)
def test_predict_signal_end_to_end_non_30s_epochs(cls):
    """Raw signal -> features -> fit -> predict_signal with 10-s epochs. Up to v1.0.0
    predict_signal raised TypeError (UTC round trip before merge_annotations) and
    hard-coded 30-s bounds."""
    fs, seg = 100, 10
    ytr = np.repeat(CYCLE * 2, 6)
    sig_tr = _recording(ytr, fs, seg, 0)
    m = make(cls, fs=fs, segm_size=seg)
    X, _ = m.extract_features_bulk(list(sig_tr.reshape(len(ytr), -1)), [fs] * len(ytr))
    assert X.shape[0] == len(ytr) and np.all(np.isfinite(X))
    m.fit(X, ytr)

    yte = np.repeat(CYCLE, 5)                                 # 25 epochs, 250 s
    sig = _recording(yte, fs, seg, 1)
    sig[12 * fs * seg + 10:13 * fs * seg - 10] = np.nan       # epoch 12 mostly missing
    df = m.predict_signal(sig, fs)

    assert list(df.columns) == ['annotation', 'start', 'end', 'duration']
    np.testing.assert_allclose(df['end'] - df['start'], df['duration'])
    assert np.all(np.mod(df['start'], seg) == 0) and np.all(np.mod(df['end'], seg) == 0)
    assert df['start'].min() == 0 and df['end'].max() == len(yte) * seg
    # the skipped epoch leaves a gap [120, 130) s
    covered = np.zeros(len(yte), bool)
    for _, r in df.iterrows():
        covered[int(r.start // seg):int(r.end // seg)] = True
    assert not covered[12] and covered.sum() == len(yte) - 1
    # label agreement per epoch
    pred = np.full(len(yte), '', dtype=object)
    for _, r in df.iterrows():
        pred[int(r.start // seg):int(r.end // seg)] = r.annotation
    assert np.mean(pred[covered] == yte[covered]) > 0.9


def test_sleep_classifier_wrapper_train_predict():
    fs = 250
    y = np.repeat(STATES, 10)
    labels = np.r_[y, y, ['N1', 'UNKNOWN']]                   # N1 / UNKNOWN must be dropped
    Xw = np.stack([_epoch(s if s in _FREQ else 'N2', fs, 30, np.random.default_rng(i))
                   for i, s in enumerate(labels)])
    df = pd.DataFrame({'annotation': labels, 'freq': [0] * len(y) + [72.5] * len(y) + [0, 0]})
    w = C.SleepClassifierWrapper(n_jobs=1)
    w.train(Xw, df, fs=fs)
    assert sorted(w.MODEL) == [0, 2, 7, 72.5]
    for mdl in w.MODEL.values():
        assert set(mdl.STATES) == set(STATES)

    yt = np.repeat(['AWAKE', 'N2', 'N3', 'N2', 'REM'], 3)
    r = np.random.default_rng(99)
    s500 = np.concatenate([_epoch(s, 500, 30, r) for s in yt])
    s500[100:200] = np.nan
    before = s500.copy()
    out = w.predict_signal(s500, 500, 0)
    np.testing.assert_array_equal(s500, before)              # caller's array untouched
    assert list(out['annotation']) == ['AWAKE', 'N2', 'N3', 'N2', 'REM']
    np.testing.assert_allclose(out['duration'], 90.0)

    with pytest.raises(ValueError, match='stim_freq'):
        w.predict_signal(s500[::2], 250, 5)


def test_scores_to_annotations_uses_segment_size():
    sc = pd.DataFrame({'A': [1, 1, 0, 0], 'B': [0, 0, 1, 1]})
    df = C._scores_to_annotations(sc, np.array([0, 10, 20, 30]), 10)
    assert df.to_dict('list') == {'annotation': ['A', 'B'], 'start': [0.0, 20.0],
                                  'end': [20.0, 40.0], 'duration': [20.0, 20.0]}
