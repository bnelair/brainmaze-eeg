import numpy as np
import pytest

from brainmaze_eeg.spikes import detect_spikes_barkmeier, DEFAULT_THRESHOLDS

FS = 512


def _biphasic(x, center, amp=600.0):
    """Add a sharp biphasic spike: central lobe with flanking troughs, into x in place."""
    w = int(0.02 * FS)
    k = np.arange(-w, w + 1)
    lobe = amp * np.exp(-(k / (0.004 * FS)) ** 2)
    troughs = (0.33 * amp * np.exp(-((k - 0.02 * FS) / (0.004 * FS)) ** 2)
               + 0.33 * amp * np.exp(-((k + 0.02 * FS) / (0.004 * FS)) ** 2))
    seg = lobe - troughs
    s = x[center - w:center + w + 1]
    s += seg[:len(s)]


def _noise(n, seed=0, sd=20.0):
    return np.random.default_rng(seed).normal(0, sd, n)


def test_detects_clean_positive_spikes_without_duplicates():
    x = _noise(10 * FS)
    times = [1, 3, 5, 7, 9]
    for t in times:
        _biphasic(x, int(t * FS))
    out = detect_spikes_barkmeier(x, FS)
    assert len(out) == len(times)
    got = sorted(d['peak_time'] for d in out)
    np.testing.assert_allclose(got, times, atol=0.03)


def test_detects_negative_going_spike():
    # regression: candidate detection must use |narrow band|, not narrow band > 0
    x = _noise(4 * FS, seed=1)
    _biphasic(x, int(2 * FS), amp=-600.0)
    assert len(detect_spikes_barkmeier(x, FS)) == 1


def test_pure_noise_false_positive_rate_is_bounded():
    # Block-scaling normalises the channel's median amplitude to `scale`, so on an
    # isolated single channel it inflates the noise floor and a fixed threshold fires at
    # a low but non-zero rate. This is a property of the Barkmeier method (its block
    # scaling is designed to be MULTICHANNEL); documented in the module. Here we only pin
    # that the rate stays low, not zero.
    out = detect_spikes_barkmeier(_noise(20 * FS, seed=2), FS)
    assert len(out) / 20.0 < 1.0   # fewer than 1 false positive per second


def test_high_threshold_rejects_everything_conjunctive_and():
    # the acceptance test is a single conjunctive AND: raising any one threshold out of
    # reach must reject the spike. A spurious "all-below" acceptance branch (the bug this
    # replaces) would instead resurrect it.
    x = _noise(4 * FS, seed=4)
    _biphasic(x, int(2 * FS), amp=600.0)
    huge = {'total_amp': 1e9, 'slope': DEFAULT_THRESHOLDS['slope'], 'half_dur': DEFAULT_THRESHOLDS['half_dur']}
    assert detect_spikes_barkmeier(x, FS, thresholds=huge) == []
    huge = {'total_amp': DEFAULT_THRESHOLDS['total_amp'], 'slope': 1e12, 'half_dur': DEFAULT_THRESHOLDS['half_dur']}
    assert detect_spikes_barkmeier(x, FS, thresholds=huge) == []


def test_refractory_is_in_seconds():
    # two discharges 120 ms apart: kept with no refractory, merged with a 300 ms one
    x = _noise(4 * FS, seed=3)
    _biphasic(x, int(1.0 * FS))
    _biphasic(x, int(1.0 * FS) + int(0.12 * FS))
    assert len(detect_spikes_barkmeier(x, FS, refractory=0.0)) == 2
    assert len(detect_spikes_barkmeier(x, FS, refractory=0.30)) == 1


def test_multichannel_block_scaling_preserves_amplitude_ratio():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 20, 6 * FS), rng.normal(0, 20, 6 * FS)])
    _biphasic(X[0], int(3 * FS), amp=800.0)
    _biphasic(X[1], int(3 * FS), amp=300.0)
    out = detect_spikes_barkmeier(X, FS)
    amp = {d['channel']: d['total_amp'] for d in out}
    assert 0 in amp and 1 in amp
    assert amp[0] / amp[1] == pytest.approx(800 / 300, rel=0.25)


def test_1d_and_2d_shapes():
    x = _noise(4 * FS, seed=6)
    _biphasic(x, int(2 * FS))
    out1d = detect_spikes_barkmeier(x, FS)
    assert all(d['channel'] == 0 for d in out1d)

    X = np.vstack([x, _noise(4 * FS, seed=7)])
    out2d = detect_spikes_barkmeier(X, FS)
    assert set(d['channel'] for d in out2d) <= {0, 1}


def test_spike_near_end_does_not_crash():
    x = _noise(4 * FS, seed=8)
    _biphasic(x, x.size - int(0.01 * FS))
    detect_spikes_barkmeier(x, FS)   # must not raise


def test_detection_fields_are_consistent():
    x = _noise(6 * FS, seed=9)
    _biphasic(x, int(3 * FS))
    d = detect_spikes_barkmeier(x, FS)[0]
    assert d['total_amp'] == pytest.approx(d['left_amp'] + d['right_amp'])
    assert d['left_slope'] == pytest.approx(d['left_amp'] / d['left_dur'])
    assert d['peak_time'] == pytest.approx(d['peak_index'] / FS)


@pytest.mark.parametrize('bad_band', [(0, 50), (20, 300), (50, 20)])
def test_invalid_band_raises(bad_band):
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(_noise(FS), FS, narrow_band=bad_band)


def test_non_1d_2d_input_raises():
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(np.zeros((2, 3, FS)), FS)


def test_default_thresholds_not_mutated_by_call():
    before = dict(DEFAULT_THRESHOLDS)
    detect_spikes_barkmeier(_noise(2 * FS), FS, thresholds={'total_amp': 1, 'slope': 1, 'half_dur': 0})
    assert DEFAULT_THRESHOLDS == before


def test_partial_thresholds_dict_merges_with_defaults():
    # review #61: a partial dict must fill from defaults, not raise a later KeyError
    x = _noise(6 * FS, seed=11)
    _biphasic(x, int(3 * FS))
    out = detect_spikes_barkmeier(x, FS, thresholds={'total_amp': 600.0})  # slope/half_dur from defaults
    assert isinstance(out, list)


def test_unknown_threshold_key_raises():
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(_noise(2 * FS), FS, thresholds={'TAMP': 600})  # wrong key name


# =========================================================================================
# Paper conformance (Barkmeier 2012), blocks, artifact channels, gaps, validation
# =========================================================================================
import warnings

from brainmaze_eeg.spikes import design_barkmeier_filters
from brainmaze_eeg.tests.spike_synth import pink_background, synth_ieeg


def _peaks(dets, ch=None):
    return np.array([d['peak_index'] for d in dets if ch is None or d['channel'] == ch])


def _hits(det, truth, tol):
    return sum(bool(np.any(np.abs(det - t) <= tol)) for t in truth)


def test_default_bands_are_the_papers():
    import inspect
    sig = inspect.signature(detect_spikes_barkmeier).parameters
    assert sig['broad_band'].default == (1.0, 35.0)
    assert sig['broad_order'].default == 2
    assert sig['narrow_band'].default == (20.0, 50.0)
    assert sig['block_s'].default == 60.0
    assert sig['artifact_sd'].default == 10.0
    assert sig['scale'].default == 70.0 and sig['std_coeff'].default == 4.0
    assert DEFAULT_THRESHOLDS == {'total_amp': 600.0, 'slope': 7000.0, 'half_dur': 0.010}


@pytest.mark.parametrize('fs', [200, 500, 2048])
def test_sensitivity_and_noise_rate_on_synthetic_ieeg(fs):
    x, truth = synth_ieeg(fs, dur=120.0, seed=2, n_spikes=40, amp_range=(300, 500),
                          mains_hz=None)
    det = _peaks(detect_spikes_barkmeier(x, fs))
    assert _hits(det, truth, 0.03 * fs) >= 0.9 * truth.size
    # pure 1/f background: ~0.02 false positives / s with the paper's 1-35 Hz band
    noise = pink_background(int(120 * fs), fs, 30.0, np.random.default_rng(9))
    assert len(detect_spikes_barkmeier(noise, fs)) / 120.0 < 0.08


def test_valid_mask_excludes_samples_from_block_statistics():
    fs = 500
    rng = np.random.default_rng(0)
    X = np.vstack([pink_background(60 * fs, fs, 40.0, rng) for _ in range(4)])
    _, ref = detect_spikes_barkmeier(X, fs, return_info=True)
    Y = X.copy()
    Y[1, 10 * fs:20 * fs] += rng.normal(0, 400.0, 10 * fs)   # garbage flagged invalid
    valid = np.ones(X.shape, bool)
    valid[1, 10 * fs:20 * fs] = False
    _, inf_v = detect_spikes_barkmeier(Y, fs, valid=valid, return_info=True)
    with pytest.warns(UserWarning, match='artifact'):
        _, inf_n = detect_spikes_barkmeier(Y, fs, return_info=True)
    # with the mask, channel 1's statistics come from its real samples only; without it
    # the garbage dominates its slope and the artifact rule drops the channel
    assert not inf_v['artifact'].any()
    assert inf_v['candidate_threshold'][0, 1] == pytest.approx(
        ref['candidate_threshold'][0, 1], rel=0.3)
    assert inf_n['artifact'][0, 1]
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(X, fs, valid=valid[:, :100])


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_raw_detector_raises_on_non_finite(bad):
    # regression: one NaN used to silently disable block scaling for every channel
    x = _noise(4 * FS)
    x[100] = bad
    with pytest.raises(ValueError, match='GapAwareSpikeDetector'):
        detect_spikes_barkmeier(x, FS)


def test_barkmeier_detector_object():
    from brainmaze_eeg.spikes import BarkmeierDetector
    rng = np.random.default_rng(1)
    X = np.vstack([pink_background(30 * FS, FS, 40.0, rng) for _ in range(3)])
    _biphasic(X[2], 10 * FS, 800.0)
    per_ch = BarkmeierDetector(refractory=0.1).detect(X, FS)
    flat = detect_spikes_barkmeier(X, FS, refractory=0.1)
    assert len(per_ch) == 3
    assert [d for ch in per_ch for d in ch] == flat
    assert all(d['channel'] == c for c, ch in enumerate(per_ch) for d in ch)
    with pytest.raises(TypeError):
        BarkmeierDetector(valid=None)


def test_inf_raises():
    x = _noise(4 * FS)
    x[100] = -np.inf
    with pytest.raises(ValueError, match='inf'):
        detect_spikes_barkmeier(x, FS)


@pytest.mark.parametrize('shape', [(5000, 4), (5000, 40)])
def test_transposed_input_raises(shape):
    with pytest.raises(ValueError, match='transpose'):
        detect_spikes_barkmeier(np.zeros(shape), 500)


def test_record_shorter_than_one_second_raises():
    with pytest.raises(ValueError, match='< 1 s'):
        detect_spikes_barkmeier(np.zeros(100), 500)


@pytest.mark.parametrize('dur,expect', [(170, [[0, 60], [60, 120], [120, 170]]),
                                        (140, [[0, 60], [60, 140]]),
                                        (45, [[0, 45]])])
def test_one_minute_blocks(dur, expect):
    fs = 200
    _, info = detect_spikes_barkmeier(_noise(dur * fs, seed=1), fs, return_info=True)
    np.testing.assert_array_equal(info['blocks'], np.array(expect) * fs)
    assert info['scale_factor'].shape == (len(expect),)
    _, info = detect_spikes_barkmeier(_noise(dur * fs, seed=1), fs, block_s=None,
                                      return_info=True)
    np.testing.assert_array_equal(info['blocks'], [[0, dur * fs]])


def test_blocks_adapt_to_a_change_of_background_amplitude():
    # 2 min quiet (30 uV) then 2 min loud (90 uV): with one-minute blocks the scaling follows
    # the background, so the noise detection rate is similar in both halves; with one
    # whole-record block the loud half is over-scaled and floods with false positives.
    fs = 500
    rng = np.random.default_rng(3)
    X = np.vstack([pink_background(240 * fs, fs, 1.0, rng) for _ in range(8)])
    X[:, :120 * fs] *= 30.0
    X[:, 120 * fs:] *= 90.0
    half = 120 * fs

    def rates(dets):
        p = _peaks(dets)
        return np.sum(p < half) / (8 * 120.0), np.sum(p >= half) / (8 * 120.0)

    q_b, l_b = rates(detect_spikes_barkmeier(X, fs))
    q_w, l_w = rates(detect_spikes_barkmeier(X, fs, block_s=None))
    assert l_b < 0.1 and q_b < 0.1
    assert l_w > 5 * max(l_b, 0.01)


def test_artifact_channel_rule():
    fs = 500
    rng = np.random.default_rng(4)
    X = np.vstack([pink_background(120 * fs, fs, 40.0, rng) for _ in range(8)])
    X[5, 60 * fs:] += rng.normal(0, 2000.0, 60 * fs)          # artifact in minute 2 only
    with pytest.warns(UserWarning, match='artifact'):
        dets, info = detect_spikes_barkmeier(X, fs, return_info=True)
    np.testing.assert_array_equal(info['artifact'][:, 5], [False, True])
    assert info['artifact'].sum() == 1
    assert not any(d['channel'] == 5 and d['block'] == 1 for d in dets)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _, info = detect_spikes_barkmeier(X, fs, artifact_sd=None, return_info=True)
    assert not info['artifact'].any()


def test_scale_invariance():
    fs = 500
    x, _ = synth_ieeg(fs, dur=60.0, seed=4, amp_range=(300, 500), mains_hz=None)
    a = _peaks(detect_spikes_barkmeier(x, fs))
    b = _peaks(detect_spikes_barkmeier(x * 1e-6, fs))
    np.testing.assert_array_equal(a, b)


def test_merge_happens_before_refractory():
    # two candidates 40 ms apart (merged: largest kept) and a third 100 ms later; with a
    # 60 ms refractory the merge must pick the largest of the cluster first
    x = _noise(4 * FS, seed=12, sd=5.0)
    _biphasic(x, int(1.0 * FS), amp=500.0)
    _biphasic(x, int(1.04 * FS), amp=900.0)
    dets = detect_spikes_barkmeier(x, FS, refractory=0.06)
    big = max(dets, key=lambda d: d['total_amp'])
    assert abs(big['peak_index'] - int(1.04 * FS)) <= 0.005 * FS


@pytest.mark.parametrize('kw', [dict(broad_band=(1, 300)), dict(narrow_band=(50, 20)),
                                dict(narrow_order=0), dict(block_s=0), dict(block_s=-60),
                                ])
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        detect_spikes_barkmeier(_noise(4 * FS), FS, **kw)


def test_info_filters_are_the_designed_ones():
    _, info = detect_spikes_barkmeier(_noise(4 * FS), FS, return_info=True)
    ref = design_barkmeier_filters(FS)
    np.testing.assert_array_equal(info['filters']['broad'], ref['broad'])
    np.testing.assert_array_equal(info['filters']['narrow'], ref['narrow'])


def test_artifact_rule_with_zero_variance_reference_channels():
    from brainmaze_eeg.spikes.barkmeier import _artifact_channels
    flags = _artifact_channels(np.array([1.0, 1.0, 1.0, 5.0]), np.ones(4, bool), 10.0)
    np.testing.assert_array_equal(flags, [False, False, False, True])
    flags = _artifact_channels(np.array([1.0, 1.0, 1.0, 1.0]), np.ones(4, bool), 10.0)
    assert not flags.any()

