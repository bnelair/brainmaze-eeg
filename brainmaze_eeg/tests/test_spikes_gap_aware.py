"""GapAwareSpikeDetector: the gap-handling layer around the raw spike detectors."""

import numpy as np
import pytest

from brainmaze_eeg.spikes import (BarkmeierDetector, GapAwareSpikeDetector, JancaDetector,
                                  SpikeDetectorHilbert, detect_spikes_barkmeier,
                                  detect_spikes_janca)
from brainmaze_eeg.tests.spike_synth import synth_ieeg

FS = 500.0


def _montage(n_ch=3, dur=120.0, seed=0, amp=(250, 400)):
    xs = [synth_ieeg(FS, dur=dur, seed=seed + c, amp_range=amp, mains_hz=None)[0]
          for c in range(n_ch)]
    return np.vstack(xs)


DETECTORS = [JancaDetector(), BarkmeierDetector(), SpikeDetectorHilbert()]


def _idx(item):
    if isinstance(item, np.ndarray):
        return item
    return np.array([d['peak_index'] for d in item], dtype=np.int64)


@pytest.mark.parametrize('det', DETECTORS, ids=lambda d: type(d).__name__)
def test_clean_data_identical_to_raw_detector(det):
    X = _montage()
    wrapped = GapAwareSpikeDetector(det).detect(X, FS)
    raw = det.detect(X, FS)
    assert len(wrapped) == len(raw) == 3
    for w, r in zip(wrapped, raw):
        if isinstance(r, np.ndarray):
            np.testing.assert_array_equal(w, r)
        else:
            assert w == r


def test_clean_1d_matches_function_api():
    x = _montage(1)[0]
    np.testing.assert_array_equal(GapAwareSpikeDetector(JancaDetector()).detect(x, FS),
                                  detect_spikes_janca(x, FS))
    recs = GapAwareSpikeDetector(BarkmeierDetector()).detect(x, FS)
    assert recs == detect_spikes_barkmeier(x, FS)


@pytest.mark.parametrize('det', DETECTORS, ids=lambda d: type(d).__name__)
@pytest.mark.parametrize('fill', ['mirror', 'pink', 'linear'])
def test_no_detection_in_or_near_gaps_multichannel_different_gaps(det, fill):
    X = _montage(3)
    gaps = {0: [(10.0, 12.5), (60.0, 60.05)], 1: [(0.0, 3.0), (115.0, 120.0)],
            2: [(30.0, 70.0)]}
    for c, gl in gaps.items():
        for a, b in gl:
            X[c, int(a * FS):int(b * FS)] = np.nan
    X[2, 80 * int(FS)] = np.inf
    margin = 0.1
    out, info = GapAwareSpikeDetector(det, fill=fill, edge_margin_s=margin).detect(
        X, FS, return_info=True, return_mask=True)
    gaps[2].append((80.0, 80.0 + 1 / FS))
    for c, gl in gaps.items():
        t = _idx(out[c]) / FS
        for a, b in gl:
            assert not np.any((t >= a - margin) & (t < b + margin)), (c, a, b)
        np.testing.assert_allclose(info['gap_intervals_s'][c], gl, atol=1 / FS)
    assert info['gap_mask'].shape == X.shape
    np.testing.assert_array_equal(info['gap_mask'], ~np.isfinite(X))
    assert not info['all_nan'].any()
    # valid time = record minus gaps widened by the margin (clipped to the record)
    expect = [120 - (2.5 + 0.2) - (0.05 + 0.2), 120 - (3.0 + 0.1) - (5.0 + 0.1),
              120 - (40 + 0.2) - (1 / FS + 0.2)]
    np.testing.assert_allclose(info['valid_s'], expect, atol=2 / FS)
    np.testing.assert_allclose(info['valid_fraction'], np.array(expect) / 120, atol=1e-4)


def test_detections_far_from_gaps_unchanged():
    X = _montage(2)
    clean = GapAwareSpikeDetector(JancaDetector()).detect(X, FS)
    Xg = X.copy()
    Xg[0, int(50 * FS):int(55 * FS)] = np.nan
    gappy = GapAwareSpikeDetector(JancaDetector()).detect(Xg, FS)
    np.testing.assert_array_equal(gappy[1], clean[1])            # other channel untouched
    far = lambda d: d[(d < 44 * FS) | (d >= 61 * FS)]
    np.testing.assert_array_equal(far(gappy[0]), far(clean[0]))


def test_one_nan_does_not_change_other_channels_barkmeier():
    # regression for the old NaN -> no scaling for every channel bug, through the wrapper
    X = _montage(4)
    clean, = [GapAwareSpikeDetector(BarkmeierDetector()).detect(X, FS)]
    Xg = X.copy()
    Xg[3, 1234] = np.nan
    gappy = GapAwareSpikeDetector(BarkmeierDetector()).detect(Xg, FS)
    for c in range(3):
        np.testing.assert_array_equal(_idx(gappy[c]), _idx(clean[c]))


@pytest.mark.parametrize('det', DETECTORS, ids=lambda d: type(d).__name__)
def test_all_nan_channel_is_flagged_not_crashing(det):
    X = _montage(3)
    X[1] = np.nan
    with pytest.warns(RuntimeWarning, match='no finite sample'):
        out, info = GapAwareSpikeDetector(det).detect(X, FS, return_info=True)
    assert len(out[1]) == 0
    np.testing.assert_array_equal(info['all_nan'], [False, True, False])
    assert info['valid_s'][1] == 0
    # the other channels are analysed exactly as without the dead channel
    ref = GapAwareSpikeDetector(det).detect(X[[0, 2]], FS)
    for a, b in zip([out[0], out[2]], ref):
        np.testing.assert_array_equal(_idx(a), _idx(b))
    if isinstance(out[0], list):
        assert all(d['channel'] == 2 for d in out[2])


def test_every_channel_missing():
    X = np.full((2, 5000), np.nan)
    with pytest.warns(RuntimeWarning):
        out, info = GapAwareSpikeDetector(JancaDetector()).detect(X, FS, return_info=True)
    assert [len(o) for o in out] == [0, 0] and info['all_nan'].all()


def test_detector_class_plus_kwargs_and_validation():
    X = _montage(1, dur=30.0)
    a = GapAwareSpikeDetector(JancaDetector, {'threshold': 4.0}).detect(X, FS)
    b = detect_spikes_janca(X, FS, threshold=4.0)
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(TypeError):
        GapAwareSpikeDetector(object())
    with pytest.raises(TypeError):
        GapAwareSpikeDetector(JancaDetector(), {'threshold': 4.0})
    with pytest.raises(ValueError):
        GapAwareSpikeDetector(JancaDetector(), fill='zeros')
    with pytest.raises(ValueError):
        GapAwareSpikeDetector(JancaDetector(), edge_margin_s=-1)
    with pytest.raises(ValueError, match='transpose'):
        GapAwareSpikeDetector(JancaDetector()).detect(X.T, FS)


def test_custom_detector_protocol():
    class Threshold:                      # a minimal third-party detector
        def detect(self, x, fs):
            return [np.flatnonzero((np.abs(c[1:-1]) > 5) & (np.abs(c[1:-1]) >= np.abs(c[:-2]))
                                   & (np.abs(c[1:-1]) > np.abs(c[2:]))) + 1 for c in x]
    x = np.zeros(1000)
    x[[100, 500, 900]] = 10.0
    x[480:495] = np.nan
    out, info = GapAwareSpikeDetector(Threshold(), fill='linear', edge_margin_s=0.1,
                                      flat_as_gap_s=None).detect(x, 100.0, return_info=True)
    np.testing.assert_array_equal(out, [100, 900])        # 500 is 0.05 s after the gap
    assert info['n_removed'] == 1

    class Broken:
        def detect(self, x, fs):
            return np.zeros(3)
    with pytest.raises(TypeError, match='one entry per channel'):
        GapAwareSpikeDetector(Broken()).detect(np.random.default_rng(0).normal(size=(2, 100)),
                                               100.0)


def test_empty_barkmeier_channel_keeps_record_type():
    X = _montage(3, dur=60.0)
    X[1] = np.random.default_rng(0).normal(0, 1e-3, X.shape[1])   # quiet channel: no spikes
    raw = BarkmeierDetector().detect(X, FS)
    wrapped = GapAwareSpikeDetector(BarkmeierDetector()).detect(X, FS)
    assert wrapped == raw
    assert all(isinstance(w, list) for w in wrapped)
    X[2] = np.nan
    with pytest.warns(RuntimeWarning):
        wrapped = GapAwareSpikeDetector(BarkmeierDetector()).detect(X, FS)
    assert all(isinstance(w, list) for w in wrapped)


# ------------------------------------------------- round 2 (independent review of #67)
from brainmaze_eeg.spikes._gaps import mask_in_gaps     # noqa: E402


def _extra_missing(got, ref, tol):
    extra = sum(not np.any(np.abs(ref - d) <= tol) for d in got)
    missing = sum(not np.any(np.abs(got - d) <= tol) for d in ref)
    return extra, missing


def _dense_montage(seed, dur=300.0):
    return np.vstack([synth_ieeg(FS, dur=dur, seed=seed + c, n_spikes=int(dur / 2),
                                 amp_range=(150, 400), mains_hz=None)[0] for c in range(2)])


@pytest.mark.parametrize('seed', [0, 10])
def test_many_short_gaps_do_not_add_detections_in_valid_time(seed):
    """R1: 100 ms dropouts every second. The old default (0.1 s linear fill) lowered Janca's
    threshold everywhere (+4..+7 detections per channel here, +67 on real data); the defaults
    keep the valid-time detections within 2 of the gap-free run."""
    X = _dense_montage(seed)
    clean = detect_spikes_janca(X, FS)
    Xg = X.copy()
    L = int(0.1 * FS)
    for s in np.arange(int(FS), X.shape[1] - L, int(FS)):
        Xg[:, s:s + L] = np.nan

    def compare(det, info, margin):
        out = []
        for c in range(2):
            ref = clean[c][~mask_in_gaps(clean[c], info['gaps'][c], FS, units='samples', gap_units='samples',
                                         margin_s=margin)]
            out.append(_extra_missing(det[c], ref, 10))
        return out

    det, info = GapAwareSpikeDetector(JancaDetector()).detect(Xg, FS, return_info=True)
    np.testing.assert_allclose(info['valid_fraction'], 0.5, atol=0.01)   # 0.1 s gap + 2 x 0.2
    for extra, missing in compare(det, info, 0.2):
        assert extra <= 2 and missing <= 1
    old = GapAwareSpikeDetector(JancaDetector(), short_gap_s=0.1, edge_margin_s=0.1)
    det, info = old.detect(Xg, FS, return_info=True)
    assert max(e for e, _ in compare(det, info, 0.1)) >= 4      # the test can see the bias


@pytest.mark.parametrize('det', [JancaDetector(), SpikeDetectorHilbert()],
                         ids=lambda d: type(d).__name__)
def test_near_gap_detections_match_gap_free_run(det):
    """R10: compare 0.2-3 s from each gap (where fill effects live), at most 1 per gap."""
    X = _dense_montage(3, dur=240.0)
    clean = det.detect(X, FS)
    rng = np.random.default_rng(5)
    Xg = X.copy()
    gaps = []
    for c in range(2):
        for s in rng.choice(np.arange(10, 225, 9), 12, replace=False):
            L = rng.choice([0.5, 2.0, 5.0])
            Xg[c, int(s * FS):int((s + L) * FS)] = np.nan
            gaps.append((c, s, s + L))
    out, info = GapAwareSpikeDetector(det).detect(Xg, FS, return_info=True)
    for c in range(2):
        g = info['gaps'][c]
        near = (mask_in_gaps(clean[c], g, FS, units='samples', gap_units='samples', margin_s=3.0)
                & ~mask_in_gaps(clean[c], g, FS, units='samples', gap_units='samples', margin_s=0.2))
        near_out = (mask_in_gaps(out[c], g, FS, units='samples', gap_units='samples', margin_s=3.0))
        extra, missing = _extra_missing(out[c][near_out], clean[c][near], 10)
        assert extra + missing <= len(g), (c, extra, missing, len(g))


def test_constant_runs_are_gaps_by_default():
    """R4: missing data stored as a constant (not NaN)."""
    X = _montage(1, dur=120.0)[0]
    raw_clean = detect_spikes_janca(X, FS)
    x = X.copy()
    runs = [(30.0, 33.0), (70.0, 70.5), (100.0, 110.0)]
    for a, b in runs:
        x[int(a * FS):int(b * FS)] = 0.1975
    raw = detect_spikes_janca(x, FS)                     # the raw detector fires at the steps
    out, info = GapAwareSpikeDetector(JancaDetector()).detect(x, FS, return_info=True)
    np.testing.assert_allclose(info['flat_runs'] / FS, runs, atol=1 / FS)
    t = out / FS
    for a, b in runs:
        assert not np.any((t >= a - 0.2) & (t < b + 0.2))
    near_raw = sum(np.any((raw / FS >= a - 1) & (raw / FS < b + 1)) for a, b in runs)
    assert near_raw >= 1
    # outside the runs the wrapper agrees with the clean recording
    far = ~mask_in_gaps(raw_clean, info['gaps'], FS, units='samples', gap_units='samples', margin_s=3.0)
    extra, missing = _extra_missing(out, raw_clean[far], 10)
    assert missing == 0
    # disabled: the constant runs are not gaps
    _, info = GapAwareSpikeDetector(JancaDetector(), flat_as_gap_s=None).detect(
        x, FS, return_info=True)
    assert len(info['gaps']) == 0 and len(info['flat_runs']) == 0


def test_constant_channel_is_reported_as_missing():
    X = _montage(2, dur=30.0)
    X[1] = 0.0
    with pytest.warns(RuntimeWarning, match='constant throughout'):
        out, info = GapAwareSpikeDetector(BarkmeierDetector()).detect(X, FS, return_info=True)
    assert info['all_nan'][1] and out[1] == []


def test_real_signal_is_not_flat():
    # quantised real-like data: no run of equal samples reaches 0.1 s
    X = np.round(_montage(2, dur=60.0))
    _, info = GapAwareSpikeDetector(JancaDetector()).detect(X, FS, return_info=True)
    assert all(len(r) == 0 for r in info['flat_runs'])


@pytest.mark.parametrize('kw', [dict(fill_kwargs={'context_s': -1}),
                                dict(fill_kwargs={'taper_s': np.nan}),
                                dict(fill_kwargs={'beta': np.inf}), dict(seed='abc'),
                                dict(seed=-1), dict(seed=1.5), dict(short_gap_s=np.inf),
                                dict(short_gap_s=-0.1), dict(edge_margin_s=np.nan),
                                dict(flat_as_gap_s=0), dict(flat_as_gap_s=np.inf),
                                dict(fill=None)])
def test_wrapper_options_validated_at_construction(kw):
    """R6: rejected before any data is seen (previously only on the first gap)."""
    with pytest.raises((ValueError, TypeError)):
        GapAwareSpikeDetector(JancaDetector(), **kw)


def test_fill_method_is_always_passed_explicitly(monkeypatch):
    """R7: a different default in the fill module (e.g. brainmaze_utils' 'spectral') must
    not change the wrapper's fill."""
    import brainmaze_eeg.spikes.gap_aware as ga
    seen = []
    real = ga.fill_gaps

    def spy(x, fs, **kw):
        seen.append(kw)
        return real(x, fs, **kw)
    monkeypatch.setattr(ga, 'fill_gaps', spy)
    x = _montage(1, dur=30.0)
    x[0, 5000:6000] = np.nan
    GapAwareSpikeDetector(JancaDetector()).detect(x, FS)
    assert seen and seen[0]['method'] == 'mirror' and seen[0]['max_interp_s'] == 0.02


def test_low_valid_time_warns():
    X = _montage(1, dur=30.0)[0]
    for s in range(1, 29):
        X[int(s * FS):int(s * FS) + 200] = np.nan            # 0.4 s gaps every second
    with pytest.warns(RuntimeWarning, match='less than half'):
        _, info = GapAwareSpikeDetector(JancaDetector()).detect(X, FS, return_info=True)
    assert info['valid_fraction'] < 0.5


def test_defaults():
    import inspect
    p = inspect.signature(GapAwareSpikeDetector).parameters
    assert p['short_gap_s'].default == 0.02 and p['edge_margin_s'].default == 0.2
    assert p['fill'].default == 'mirror' and p['flat_as_gap_s'].default == 0.1


@pytest.mark.parametrize('det', DETECTORS, ids=lambda d: type(d).__name__)
def test_float32_input_and_channel_at_a_time_path(det):
    """R8: float32 input is converted per channel; channel-independent detectors are fed one
    channel at a time and give the raw multichannel result."""
    X = _montage(3).astype(np.float32)
    X[1, 4000:5000] = np.nan
    a = GapAwareSpikeDetector(det).detect(X, FS)
    b = GapAwareSpikeDetector(det).detect(X.astype(np.float64), FS)
    for p, q in zip(a, b):
        np.testing.assert_array_equal(_idx(p), _idx(q))
