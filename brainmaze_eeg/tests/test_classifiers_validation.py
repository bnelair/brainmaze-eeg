"""
End-to-end tests of every classifier in ``brainmaze_eeg.classifiers`` (issue #57, PR #69).

Each model is fitted on small synthetic data with known, separable class structure and
checked for: output shapes and label sets, accuracy well above chance on held-out data,
determinism (refitting gives identical output), no train/test leakage in the
normalisation, invariance to feature units, out-of-distribution flagging, and gap
handling. Raw-signal tests run the full ``extract_features_bulk -> fit ->
predict_signal`` path. These are wiring/numerics checks on synthetic data, not a
validation of sleep-staging accuracy on real recordings.

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
    np.testing.assert_allclose(m.Scaler.mean_, Xb.mean(axis=0), atol=1e-10)
    Zb = m.PCA.transform(m.SELECTOR.transform(m.Scaler.transform(Xb)))
    np.testing.assert_allclose(m.ZScore.mean, Zb.mean(axis=0, keepdims=True), atol=1e-10)
    np.testing.assert_allclose(m.ZScore.std, Zb.std(axis=0, keepdims=True), rtol=1e-10)

    mean0, std0 = m.ZScore.mean.copy(), m.ZScore.std.copy()
    comp0 = m.PCA.components_.copy()
    smean0 = m.Scaler.mean_.copy()
    m.predict(Xte + 50.0)           # wildly shifted test data
    np.testing.assert_array_equal(m.Scaler.mean_, smean0)
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


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize('shift', [100.0, 1e3])
def test_out_of_distribution_epoch_is_unknown(cls, shift, feature_data):
    """Review R2 of PR #69 (maintainer decision): an artefact epoch far from every state
    must not get a confident label. Up to v1.0.0 it gave 0/0 = NaN and ``predict``
    raised; round 1 of the PR gave probability 1.0 for one state."""
    Xtr, ytr, Xte, yte = feature_data
    m = make(cls)
    m.fit(Xtr, ytr)
    assert np.isfinite(m.log_lik_floor_)
    assert m.log_lik_floor_ < np.min(m.train_max_log_lik_)
    X = Xte.copy()
    X[20] += shift                                  # the reviewer's artefact case
    sc = m.scores(X).to_numpy(dtype=float)
    assert np.all(np.isnan(sc[20]))
    assert m.max_log_lik_[20] < m.log_lik_floor_
    others = np.r_[0:20, 21:len(X)]
    assert np.all(np.isfinite(sc[others]))
    np.testing.assert_allclose(sc[others].sum(axis=1), 1.0, atol=1e-9)
    pred = m.predict(X)
    assert pred[20] == C.UNKNOWN_LABEL
    assert np.mean(pred[others] == yte[others]) > 0.85
    np.testing.assert_allclose(m.max_log_likelihood(X), m.max_log_lik_)


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_log_lik_floor_configurable(cls, feature_data):
    Xtr, ytr, Xte, _ = feature_data
    X = Xte.copy()
    X[20] += 1e3
    off = make(cls, log_lik_floor=None)
    off.fit(Xtr, ytr)
    assert off.log_lik_floor_ == -np.inf
    sc = off.scores(X).to_numpy(dtype=float)
    assert np.all(np.isfinite(sc))                  # legacy behaviour: no flagging
    fixed = make(cls, log_lik_floor=1e9)            # absurd floor: everything flagged
    fixed.fit(Xtr, ytr)
    assert np.all(fixed.predict(Xte) == C.UNKNOWN_LABEL)
    with pytest.raises(ValueError, match='log_lik_floor'):
        make(cls, log_lik_floor='bogus').fit(Xtr, ytr)


def test_held_out_in_distribution_epochs_not_flagged():
    """False-flag rate of the default floor on held-out in-distribution data
    (large sample; the PR #69 notes give the full evidence)."""
    ytr = cyc_hypnogram(40, 10)
    yte = cyc_hypnogram(250, 11)                    # ~2000 held-out epochs
    Xtr, Xte = features(ytr, seed=10), features(yte, seed=11)
    for cls in (C.KDEBayesianModel, C.MVGaussBayesianModel, C.MultiChannelMVGaussBayesClassifier):
        m = make(cls, window_smooth_n=1) if cls is not C.MultiChannelMVGaussBayesClassifier else make(cls)
        m.fit(Xtr, ytr)
        flagged = np.mean(m.predict(Xte) == C.UNKNOWN_LABEL)
        assert flagged == 0.0, (cls.__name__, flagged)


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
    with pytest.warns(UserWarning, match='AWAKE'):
        m = C.SleepStructureClassifier().fit(X, y)    # default states include AWAKE/N1/REM
    assert m.STATES == ['N2', 'N3']
    assert set(m.predict(X)) <= {'N2', 'N3'}


def test_sleep_structure_classifier_rejects_unknown_labels():
    # review R3: a label outside the state list (e.g. AWAKE when the default was WAKE)
    # was silently dropped
    y = np.repeat(['ARTIFACT', 'N2', 'N3'], 6)
    X = features(y, d=3, seed=0)
    with pytest.raises(ValueError, match='ARTIFACT'):
        C.SleepStructureClassifier().fit(X, y)


def test_sleep_structure_classifier_refit_restores_states():
    # Copilot / review R3: the state list must not shrink permanently
    m = C.SleepStructureClassifier(states=STATES)
    y1 = np.repeat(['N2', 'N3'], 6)
    with pytest.warns(UserWarning):
        m.fit(features(y1, d=3, seed=0), y1)
    assert m.STATES == ['N2', 'N3']
    y2 = np.repeat(STATES, 6)
    m.fit(features(y2, d=3, seed=1), y2)
    assert m.STATES == STATES


def test_sleep_structure_classifier_too_few_pairs_clear_error():
    # Copilot: 2 epochs per state passed the old check but gaussian_kde then raised
    y = np.repeat(['N2', 'N3'], 2)
    with pytest.raises(ValueError, match='needs at least'):
        C.SleepStructureClassifier(states=['N2', 'N3']).fit(features(y, d=3, seed=0), y)


def test_sleep_structure_classifier_single_epoch_raises():
    # review R6: one epoch has no pairs -> NaN row before
    y = np.repeat(STATES, 6)
    m = C.SleepStructureClassifier(states=STATES).fit(features(y, d=3, seed=0), y)
    with pytest.raises(ValueError, match='>= 2 epochs'):
        m.scores(features(y, d=3, seed=1)[:1])


def test_markov_filter_refit_with_more_states():
    # Copilot: after a fit on N2/N3 a refit including AWAKE raised (mutated STATES)
    mf = C.SleepStageProbabilityMarkovChainFilter()
    y1 = np.array(['N2', 'N3'] * 20)
    mf.fit(pd.DataFrame({'N2': np.r_[[0.8, 0.2] * 20], 'N3': np.r_[[0.2, 0.8] * 20]}), y1)
    assert list(mf.STATES) == ['N2', 'N3']
    y2 = np.array(['AWAKE', 'N2', 'N3'] * 10)
    sc = pd.DataFrame({'AWAKE': np.r_[[0.8, 0.1, 0.1] * 10], 'N2': np.r_[[0.1, 0.8, 0.1] * 10],
                       'N3': np.r_[[0.1, 0.1, 0.8] * 10]})
    mf.fit(sc, y2)
    assert list(mf.STATES) == ['AWAKE', 'N2', 'N3']
    fresh = C.SleepStageProbabilityMarkovChainFilter()
    fresh.fit(sc, y2)
    np.testing.assert_allclose(mf.tmat, fresh.tmat)


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


def _epoch(state, fs, dur, r, noise_fs=None):
    """``noise_fs``: scale the white noise to the spectral density it has at ``noise_fs``
    (so recordings at different rates have the same in-band noise level)."""
    t = np.arange(int(round(fs * dur))) / fs
    f1, f2, a1 = _FREQ[state]
    noise = 1.0 if noise_fs is None else np.sqrt(fs / noise_fs)
    return (a1 * np.sin(2 * np.pi * f1 * t + r.uniform(0, 2 * np.pi))
            + 0.7 * np.sin(2 * np.pi * f2 * t + r.uniform(0, 2 * np.pi))
            + noise * r.normal(size=t.size))


def _recording(labels, fs, dur, seed, noise_fs=None):
    r = np.random.default_rng(seed)
    return np.concatenate([_epoch(s, fs, dur, r, noise_fs) for s in labels])


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
    s250 = np.concatenate([_epoch(s, fs, 30, r) for s in yt])
    s250[100:200] = np.nan
    before = s250.copy()
    out = w.predict_signal(s250, fs, 0)
    np.testing.assert_array_equal(s250, before)              # caller's array untouched
    assert list(out['annotation']) == ['AWAKE', 'N2', 'N3', 'N2', 'REM']
    np.testing.assert_allclose(out['duration'], 90.0)

    with pytest.raises(ValueError, match='stim_freq'):
        w.predict_signal(s250, 250, 5)

    # Copilot: a second train without 72.5-Hz epochs must not keep the old 72.5 model
    keep = df['freq'].to_numpy() != 72.5
    w.train(Xw[keep], df.loc[keep].reset_index(drop=True), fs=fs)
    assert sorted(w.MODEL) == [0, 2, 7]
    with pytest.raises(ValueError, match='stim_freq'):
        w.predict_signal(s250, fs, 72.5)


def test_sleep_classifier_wrapper_same_resampling_in_train_and_predict():
    """Review R9: any fs, and train / predict_signal resample identically (both through
    the model's checked ``extract_features_bulk``): a 500-Hz model input gives the same
    labels as in training."""
    fs = 500
    y = np.repeat(STATES, 10)
    Xw = np.stack([_epoch(s, fs, 30, np.random.default_rng(i)) for i, s in enumerate(y)])
    df = pd.DataFrame({'annotation': y, 'freq': [0] * len(y)})
    w = C.SleepClassifierWrapper(n_jobs=1)
    w.train(Xw, df, fs=fs)
    yt = np.repeat(['AWAKE', 'N2', 'N3', 'N2', 'REM'], 3)
    r = np.random.default_rng(98)
    out = w.predict_signal(np.concatenate([_epoch(s, fs, 30, r) for s in yt]), fs, 0)
    assert list(out['annotation']) == ['AWAKE', 'N2', 'N3', 'N2', 'REM']


def test_scores_to_annotations_uses_segment_size():
    sc = pd.DataFrame({'A': [1, 1, 0, 0], 'B': [0, 0, 1, 1]})
    df = C._scores_to_annotations(sc, np.array([0, 10, 20, 30]), 10)
    assert df.to_dict('list') == {'annotation': ['A', 'B'], 'start': [0.0, 20.0],
                                  'end': [20.0, 40.0], 'duration': [20.0, 20.0]}


# --------------------------------------------------------------------------- review round 2 (PR #69)
def _scaled(X, scales):
    return X * np.asarray(scales)[None, :]


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.KDEBayesianModelNC, C.MVGaussBayesianModel],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize('selector2', [True, False])
def test_invariant_to_feature_units(cls, selector2, feature_data):
    """Review R4: the real features mix Hz, [0, 1] ratios and log10 ratios. With the
    default ``standardize=True`` a per-feature rescaling of the input (which leaves the
    Bayes accuracy unchanged) must not change the predictions. Without it the
    discriminative features x 0.1 gave chance-level accuracy (or a ValueError)."""
    Xtr, ytr, Xte, yte = feature_data
    scale_sets = [np.r_[[0.1] * 4, [1.0] * 6],                                 # discriminative x 0.1
                  np.random.default_rng(5).uniform(0.1, 10, Xtr.shape[1]),     # random units
                  np.r_[[1000.0] * 4, [0.01] * 6]]
    ref = make(cls, Selector2=selector2)
    ref.fit(Xtr, ytr)
    p_ref = ref.predict(Xte)
    assert np.mean(p_ref == yte) > 0.85
    for sc in scale_sets:
        m = make(cls, Selector2=selector2)
        m.fit(_scaled(Xtr, sc), ytr)
        p = m.predict(_scaled(Xte, sc))
        assert np.mean(p == p_ref) > 0.98, (sc, np.mean(p == p_ref))
        assert np.mean(p == yte) > 0.85


def test_selector2_keeping_no_feature_raises_clearly(feature_data):
    # without standardisation this scaling made SelectFromModel keep 0 features and
    # gaussian_kde fail on an empty array
    Xtr, ytr, _, _ = feature_data
    Xs = _scaled(Xtr, np.r_[[0.1] * 4, [1.0] * 6])
    m = make(C.KDEBayesianModel, standardize=False, Selector2=True)
    try:
        m.fit(Xs, ytr)
    except ValueError as e:
        assert 'kept no feature' in str(e)
    else:   # if it does keep features the model must still be usable
        assert m.predict(Xs).shape == (Xs.shape[0],)


@pytest.mark.parametrize('cls', KDE_FAMILY, ids=lambda c: c.__name__)
def test_smoothing_and_filter_do_not_cross_gaps(cls, feature_data):
    """Review R5: rows after a gap in ``start_time`` (skipped epochs) are scored exactly as
    if the segment were scored on its own; the segment before the gap is unaffected too."""
    Xtr, ytr, Xte, _ = feature_data
    m = make(cls)
    m.fit(Xtr, ytr)
    seg = m.segm_size
    n1 = 40
    st = np.r_[np.arange(n1), np.arange(n1 + 5, Xte.shape[0] + 5)] * float(seg)
    full = m.scores(Xte, start_time=st).to_numpy(dtype=float)
    first = m.scores(Xte[:n1]).to_numpy(dtype=float)
    np.testing.assert_allclose(full[:n1], first, rtol=1e-12, atol=1e-12)
    if cls in (C.KDEBayesianCausalModel, C.MVGaussBayesianCausalModel):
        # the chain after the gap restarts in the arg-max state of its first epoch
        return
    second = m.scores(Xte[n1:]).to_numpy(dtype=float)
    np.testing.assert_allclose(full[n1:], second, rtol=1e-12, atol=1e-12)
    # without start_time the rows are treated as consecutive (smoothing crosses row n1)
    joined = m.scores(Xte).to_numpy(dtype=float)
    assert not np.allclose(joined[n1 - 1:n1 + 1], full[n1 - 1:n1 + 1])


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.KDEBayesianCausalModel,
                                 C.MultiChannelMVGaussBayesClassifier], ids=lambda c: c.__name__)
def test_predict_signal_empty_or_short(cls):
    """Copilot: every epoch below the datarate threshold (or a signal shorter than one
    epoch) returns an empty table instead of crashing in feature extraction."""
    fs, seg = 100, 10
    ytr = np.repeat(CYCLE * 2, 6)
    m = make(cls, fs=fs, segm_size=seg)
    X, _ = m.extract_features_bulk(list(_recording(ytr, fs, seg, 0).reshape(len(ytr), -1)), [fs] * len(ytr))
    m.fit(X, ytr)
    for sig in (np.full(5 * fs * seg, np.nan), np.zeros(fs * seg - 1), np.zeros(0)):
        df = m.predict_signal(sig, fs)
        assert list(df.columns) == ['annotation', 'start', 'end', 'duration'] and len(df) == 0
        sc = m.predict_signal_scores(sig, fs)
        assert sc.shape == (0, len(m.STATES))


def test_single_epoch_scores_clear():
    # review R6 for the KDE family: one epoch gives one finite row (or NaN = UNKNOWN)
    ytr = cyc_hypnogram(20, 0)
    m = make(C.KDEBayesianCausalModel)
    m.fit(features(ytr, seed=0), ytr)
    sc = m.scores(features(ytr, seed=1)[:1])
    assert sc.shape == (1, len(STATES)) and np.isclose(sc.to_numpy().sum(), 1.0)


# ---- R1: resampling sanity checks
def _utils_resamples_141_hz_correctly():
    try:
        C._resampling_self_test(141, 200, 141 * 30)
        return True
    except ValueError:
        return False


def test_resampling_141_hz_never_silently_wrong():
    """Review R1: with brainmaze_utils 2.0.0, 139-141 Hz input resampled to 200 Hz is
    finite but blown up (1e72-1e230) and every epoch was labelled REM without an error.
    Now: a ValueError naming brainmaze_utils (2.0.0), or correct labels (>= 3.0.0)."""
    fs_tr, fs_in, seg = 200, 141, 30
    ytr = np.repeat(CYCLE * 2, 6)
    m = make(C.KDEBayesianModel, fs=fs_tr, segm_size=seg)
    X, _ = m.extract_features_bulk(list(_recording(ytr, fs_tr, seg, 0).reshape(len(ytr), -1)), [fs_tr] * len(ytr))
    m.fit(X, ytr)
    yte = np.repeat(CYCLE, 4)
    sig = _recording(yte, fs_in, seg, 1, noise_fs=fs_tr)       # same in-band noise density
    if _utils_resamples_141_hz_correctly():
        df = m.predict_signal(sig, fs_in)
        pred = np.full(len(yte), '', dtype=object)
        for _, r in df.iterrows():
            pred[int(r.start // seg):int(r.end // seg)] = r.annotation
        assert np.mean(pred == yte) > 0.9
    else:
        with pytest.raises(ValueError, match='brainmaze_utils'):
            m.predict_signal(sig, fs_in)


@pytest.mark.parametrize('corrupt, match', [
    (lambda x: x * 1e80, 'RMS grew'),
    (lambda x: np.full_like(x, np.nan), 'finite fraction'),
    (lambda x: np.where(np.arange(x.size) == 3, np.inf, x), 'inf'),
])
def test_resample_epochs_rejects_corrupted_output(monkeypatch, corrupt, match):
    """The per-epoch checks catch a corrupting resampler even when the self-test passes."""
    real = C.unify_sampling_frequency
    state = {'n': 0}

    def fake(x, sampling_frequency, fs_new=None):
        out, fs = real(x, sampling_frequency, fs_new=fs_new)
        state['n'] += 1
        if state['n'] > 1:                       # first call = self-test, leave it intact
            out = [corrupt(np.asarray(o, float)) for o in out]
        return out, fs

    monkeypatch.setattr(C, 'unify_sampling_frequency', fake)
    sig = np.random.default_rng(0).normal(size=250 * 30)
    with pytest.raises(ValueError, match=match):
        C._resample_epochs([sig], [250], 200)


def test_resample_epochs_identity_at_model_rate():
    sig = np.random.default_rng(0).normal(size=6000)
    out, fs = C._resample_epochs([sig], [200], 200)
    assert fs == 200 and np.array_equal(out[0], sig)


def test_sleep_structure_classifier_default_wake_label_is_awake():
    m = C.SleepStructureClassifier()
    assert m.STATES == ['AWAKE', 'N1', 'N2', 'N3', 'REM']
    y = np.repeat(['WAKE', 'N2'], 6)
    with pytest.raises(ValueError, match='WAKE'):
        m.fit(features(y, d=3, seed=0), y)    # the old label is not silently accepted


# ---- round 3 (review 5385725783): V1 DC offset, V2 one bad epoch, V3 small-n floor, V4 'UNKNOWN' label
def _dc_sensitive(real):
    """A resampler whose error grows with the input's DC level, like brainmaze_utils 2.0.0
    (adds a ramp of 1e-3 x mean(x))."""
    def fake(x, sampling_frequency, fs_new=None):
        out, fs = real(x, sampling_frequency, fs_new=fs_new)
        return [np.asarray(o, float) + 1e-3 * np.nanmean(xi) * np.linspace(-1, 1, len(o))
                for o, xi in zip(out, x)], fs
    return fake


def test_resampling_self_test_includes_dc(monkeypatch):
    """V1: the self-test probe carries a large DC offset, so a DC-sensitive resampler is
    caught if the data path did not remove the mean."""
    monkeypatch.setattr(C, 'unify_sampling_frequency', _dc_sensitive(C.unify_sampling_frequency))
    C._resampling_self_test(250, 200, 250 * 30)                  # demeaned path: passes

    def no_demean(list_of_signals, fsamp_list, fs_new):
        out, fs = C.unify_sampling_frequency([np.asarray(s, float) for s in list_of_signals],
                                             sampling_frequency=list(fsamp_list), fs_new=fs_new)
        return [np.asarray(o, float) for o in out], fs
    monkeypatch.setattr(C, '_demeaned_resample', no_demean)
    with pytest.raises(ValueError, match='distorts'):
        C._resampling_self_test(250, 200, 250 * 30)


def test_resampled_epochs_are_demeaned_and_features_dc_invariant(monkeypatch):
    """V1: each epoch's mean is removed before resampling; the features do not depend on
    the DC level, so a DC-sensitive resampler cannot change them."""
    m = make(C.KDEBayesianModel, fs=200, segm_size=30)
    r = np.random.default_rng(0)
    ep = _epoch('N2', 250, 30, r)
    f_ref = m.extract_features(_epoch('N2', 200, 30, np.random.default_rng(0)))
    # at the model rate (no resampling) the features ignore the DC level
    e200 = _epoch('N2', 200, 30, np.random.default_rng(0))
    np.testing.assert_allclose(m.extract_features(e200 + 1e5 * e200.std()), f_ref, rtol=1e-6)
    monkeypatch.setattr(C, 'unify_sampling_frequency', _dc_sensitive(C.unify_sampling_frequency))
    x0, _ = m.extract_features_bulk([ep], [250])
    x1, _ = m.extract_features_bulk([ep + 1e5 * ep.std()], [250])
    np.testing.assert_allclose(x1, x0, rtol=1e-6)
    out, _ = C._resample_epochs([ep + 1e5 * ep.std()], [250], 200)
    assert abs(np.nanmean(out[0])) < 1e-6 * ep.std() * 1e5


@pytest.mark.parametrize('dc', [1e3, 1e5])
def test_predict_signal_dc_offset_does_not_change_labels(dc):
    """V1 end to end with the installed brainmaze_utils: a DC offset of ``dc`` x RMS at
    1400 Hz (with utils 2.0.0 and no demeaning: 31 % resampling error at 1e3 that passed
    the guard and changed 10 % of the labels silently)."""
    fs_tr, fs_in, seg = 200, 1400, 30
    ytr = np.repeat(CYCLE * 2, 6)
    m = make(C.KDEBayesianModel, fs=fs_tr, segm_size=seg)
    X, _ = m.extract_features_bulk(list(_recording(ytr, fs_tr, seg, 0).reshape(len(ytr), -1)), [fs_tr] * len(ytr))
    m.fit(X, ytr)
    yte = np.repeat(CYCLE, 3)
    sig = _recording(yte, fs_in, seg, 1, noise_fs=fs_tr)
    lab0 = C._labels_from_scores(m.predict_signal_scores(sig, fs_in))
    lab1 = C._labels_from_scores(m.predict_signal_scores(sig + dc * sig.std(), fs_in))
    assert np.array_equal(lab0, lab1)
    assert np.mean(lab0 == yte) > 0.9


def _fit_signal_model(cls, fs=100, seg=10):
    ytr = np.repeat(CYCLE * 2, 6)
    m = make(cls, fs=fs, segm_size=seg)
    X, _ = m.extract_features_bulk(list(_recording(ytr, fs, seg, 0).reshape(len(ytr), -1)), [fs] * len(ytr))
    m.fit(X, ytr)
    return m


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.KDEBayesianCausalModel], ids=lambda c: c.__name__)
def test_one_bad_resampled_epoch_does_not_abort_predict_signal(monkeypatch, cls):
    """V2: an epoch failing the per-epoch resampling checks is labelled UNKNOWN with a
    warning counting it; the other epochs are labelled as without it."""
    fs, seg, fs_in = 100, 10, 250
    m = _fit_signal_model(cls, fs, seg)
    yte = np.repeat(CYCLE, 4)
    sig = _recording(yte, fs_in, seg, 1, noise_fs=fs)
    ref = C._labels_from_scores(m.predict_signal_scores(sig, fs_in))
    real = C.unify_sampling_frequency

    def fake(x, sampling_frequency, fs_new=None):
        out, f = real(x, sampling_frequency, fs_new=fs_new)
        if len(x) > 1:                                  # not the self-test
            out = list(out)
            out[7] = np.full_like(np.asarray(out[7], float), np.nan)
        return out, f
    monkeypatch.setattr(C, 'unify_sampling_frequency', fake)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        sc = m.predict_signal_scores(sig, fs_in)
        df = m.predict_signal(sig, fs_in)
    lab = C._labels_from_scores(sc)
    assert lab[7] == C.UNKNOWN_LABEL and np.all(np.isnan(m.max_log_lik_[7:8]))
    others = np.arange(len(lab)) != 7
    # epochs well before the bad one (outside the smoothing reach) are unchanged; after it a
    # new run starts (smoothing / Markov filter restart), so allow a few changes there
    assert np.array_equal(lab[:5], ref[:5])
    assert np.mean(lab[others] == ref[others]) >= 0.8
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert any('1 of 20 epochs could not be processed' in s and 'finite fraction' in s for s in msgs)
    unk = df[df.annotation == C.UNKNOWN_LABEL]
    assert ((unk.start == 7 * seg) & (unk.end == 8 * seg)).sum() == 1


@pytest.mark.parametrize('cls', [C.KDEBayesianModel, C.MultiChannelMVGaussBayesClassifier],
                         ids=lambda c: c.__name__)
def test_flat_epoch_with_nan_features_is_unknown_not_an_error(cls):
    """V2: an exactly flat (disconnected) epoch has NaN features; up to round 2 the
    feature selector raised ValueError for the whole recording."""
    fs, seg = 100, 10
    m = _fit_signal_model(cls, fs, seg)
    yte = np.repeat(CYCLE, 4)
    sig = _recording(yte, fs, seg, 1)
    sig[5 * fs * seg:6 * fs * seg] = 3.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lab = C._labels_from_scores(m.predict_signal_scores(sig, fs))
    assert lab[5] == C.UNKNOWN_LABEL
    assert any('non-finite features' in str(x.message) for x in w)
    assert np.mean(np.delete(lab, 5) == np.delete(yte, 5)) > 0.9


def test_resample_epochs_mask_mode_reports_bad_epochs(monkeypatch):
    real = C.unify_sampling_frequency

    def fake(x, sampling_frequency, fs_new=None):
        out, f = real(x, sampling_frequency, fs_new=fs_new)
        if len(x) > 1:
            out = list(out)
            out[1] = np.asarray(out[1], float) * 1e80
        return out, f
    monkeypatch.setattr(C, 'unify_sampling_frequency', fake)
    sigs = list(np.random.default_rng(0).normal(size=(3, 250 * 30)))
    out, fs, bad = C._resample_epochs(sigs, [250] * 3, 200, on_bad='mask')
    assert list(bad) == [1] and 'RMS grew' in bad[1] and len(out) == 3
    with pytest.raises(ValueError, match='RMS grew'):
        C._resample_epochs(sigs, [250] * 3, 200)


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_auto_floor_warns_for_small_training_sets(cls, feature_data):
    """V3: below 100 training epochs the 'auto' floor is the training minimum - margin
    and gave 0.6-1.8 % false UNKNOWN; fit warns."""
    X, y, _, _ = feature_data
    small = np.concatenate([np.flatnonzero(y == s)[:20] for s in STATES])     # 80 epochs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        make(cls).fit(X[small], y[small])
    assert any("log_lik_floor='auto'" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        make(cls).fit(X, y)                                                   # ~170 epochs
        make(cls, log_lik_floor=-50.0).fit(X[small], y[small])                # fixed floor
    assert not any("log_lik_floor='auto'" in str(x.message) for x in w)


@pytest.mark.parametrize('cls', ALL_MODELS, ids=lambda c: c.__name__)
def test_fit_rejects_unknown_training_label(cls, feature_data):
    """V4: 'UNKNOWN' is the out-of-distribution label; a trained 'UNKNOWN' state would be
    indistinguishable from it."""
    X, y, _, _ = feature_data
    y = np.array(y, dtype=object)
    y[:15] = C.UNKNOWN_LABEL
    with pytest.raises(ValueError, match="'UNKNOWN'"):
        make(cls).fit(X, y)


def test_sleep_structure_classifier_rejects_unknown_label_and_state():
    y = np.repeat(['AWAKE', 'N2', 'UNKNOWN'], 6)
    with pytest.raises(ValueError, match="'UNKNOWN'"):
        C.SleepStructureClassifier().fit(features(y, d=3, seed=0), y)
    m = C.SleepStructureClassifier(states=['AWAKE', 'N2', 'UNKNOWN'])
    with pytest.raises(ValueError, match="'UNKNOWN'"):
        m.fit(features(y, d=3, seed=0), y)


def test_markov_filter_rejects_unknown_label():
    sc = pd.DataFrame(np.full((4, 2), 0.5), columns=['AWAKE', 'N2'])
    with pytest.raises(ValueError, match='UNKNOWN'):
        C.SleepStageProbabilityMarkovChainFilter().fit(sc, ['AWAKE', 'N2', 'UNKNOWN', 'N2'])
