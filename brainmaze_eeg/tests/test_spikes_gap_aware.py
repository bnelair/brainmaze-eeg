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
    out, info = GapAwareSpikeDetector(Threshold(), fill='linear').detect(x, 100.0,
                                                                         return_info=True)
    np.testing.assert_array_equal(out, [100, 900])        # 500 is 0.05 s after the gap
    assert info['n_removed'] == 1

    class Broken:
        def detect(self, x, fs):
            return np.zeros(3)
    with pytest.raises(TypeError, match='one entry per channel'):
        GapAwareSpikeDetector(Broken()).detect(np.zeros((2, 100)), 100.0)


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
