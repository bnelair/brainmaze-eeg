import numpy as np
import pytest

from brainmaze_eeg.spikes import SpikeDetectorHilbert, spike_detector_hilbert_v24

FS = 512


def _spike(x, center, amp=300.0):
    """Add a 30 Hz gamma-modulated sharp transient (an IED-like envelope bump) in place."""
    w = int(0.025 * FS)
    k = np.arange(-w, w + 1)
    x[center - w:center + w + 1] += amp * np.exp(-(k / (0.008 * FS)) ** 2) * np.cos(2 * np.pi * 30 * k / FS)


def _recording(duration_s, spikes, n_ch=1, seed=0, sd=15.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, sd, (int(duration_s * FS), n_ch))
    for ch, times in spikes.items():
        for t in times:
            _spike(x[:, ch], int(t * FS))
    return x


def _pos_by_channel(out):
    d = {}
    for p, c in zip(out['pos'], out['chan']):
        d.setdefault(int(c), []).append(round(float(p), 1))
    return {c: sorted(v) for c, v in d.items()}


def test_detects_spikes_at_known_times_and_channels():
    truth = {0: [3, 10, 20], 1: [6, 15]}
    X = _recording(30, truth, n_ch=2)
    out, *_ = SpikeDetectorHilbert().run(X, FS)
    got = _pos_by_channel(out)
    for ch, times in truth.items():
        assert got.get(ch, []) == pytest.approx(times, abs=0.1)


def test_alias_matches_class():
    assert spike_detector_hilbert_v24 is SpikeDetectorHilbert


def test_overlap_invariance_whole_vs_buffered():
    # a detection set produced in one block must equal the set produced with forced
    # multi-segment buffering -- no boundary duplicates or losses (the fixed overlap logic)
    spikes = {0: [5, 20, 35, 50, 65, 80, 95, 110]}
    X = _recording(120, spikes, n_ch=1)
    whole = SpikeDetectorHilbert(buffering=300).run(X, FS)[0]
    buffered = SpikeDetectorHilbert(buffering=20).run(X, FS)[0]
    sw = sorted(round(p, 2) for p in whole['pos'])
    sb = sorted(round(p, 2) for p in buffered['pos'])
    assert len(sw) == len(sb)
    assert sw == pytest.approx(sb, abs=0.02)


@pytest.mark.parametrize('in_fs', [500, 512, 1024, 2048])
def test_decimation_length_and_frequency(in_fs):
    det = SpikeDetectorHilbert(decimation=200)
    t = np.arange(0, 5, 1 / in_fs)
    x = np.sin(2 * np.pi * 30 * t)[:, None]
    dec = det._resample(x, in_fs, 200)
    assert dec.shape[0] == pytest.approx(5 * 200, abs=1)
    freqs = np.fft.rfftfreq(dec.shape[0], 1 / 200)
    assert freqs[np.abs(np.fft.rfft(dec[:, 0])).argmax()] == pytest.approx(30, abs=0.5)


def test_run_reports_decimated_rate_in_output_length():
    X = _recording(20, {0: [8]}, n_ch=1)
    _, _, d_decim, envelope, background, pdf = SpikeDetectorHilbert(decimation=200).run(X, FS)
    assert d_decim.shape[0] == pytest.approx(20 * 200, abs=2)
    assert envelope.shape == d_decim.shape
    assert background.shape == (d_decim.shape[0], 1, 2)
    assert pdf.shape == d_decim.shape


def test_single_channel_1d_input():
    x = _recording(20, {0: [8]}, n_ch=1)[:, 0]
    out, *_ = SpikeDetectorHilbert().run(x, FS)
    assert len(out['pos']) >= 1
    assert set(int(c) for c in out['chan']) <= {0}


def test_all_zero_channel_is_handled():
    Z = np.zeros((10 * FS, 2))
    _spike(Z[:, 0], int(5 * FS))
    out, *_ = SpikeDetectorHilbert().run(Z, FS)
    chans = set(int(c) for c in out['chan']) if len(out['pos']) else set()
    assert 1 not in chans     # the zero channel produces nothing


def test_pure_noise_low_false_positive_rate():
    X = _recording(30, {}, n_ch=1, seed=7)
    out, *_ = SpikeDetectorHilbert().run(X, FS)
    assert len(out['pos']) / 30.0 < 1.0


def test_ambiguous_markers_with_k2_above_k1():
    # k2 > k1 enables the ambiguous (0.5) class; detector must still run and may emit them
    truth = {0: [5, 12]}
    X = _recording(20, truth, n_ch=1)
    out, *_ = SpikeDetectorHilbert(k1=3.65, k2=4.5).run(X, FS)
    assert set(out['con'].tolist()) <= {1.0, 0.5}


def test_k2_below_k1_raises():
    with pytest.raises(ValueError):
        SpikeDetectorHilbert(k1=3.65, k2=3.0)


def test_beta_detection_not_implemented():
    X = _recording(10, {0: [5]}, n_ch=1)
    with pytest.raises(NotImplementedError):
        SpikeDetectorHilbert(beta=15).run(X, FS)


def test_unknown_parameter_raises():
    with pytest.raises(TypeError):
        SpikeDetectorHilbert(not_a_param=1)


def test_bandwidth_above_nyquist_raises():
    X = _recording(5, {0: [2]}, n_ch=1)
    with pytest.raises(ValueError):
        SpikeDetectorHilbert(bandwidth=[10, 150], decimation=200).run(X, FS)


def test_discharges_shapes_match_channels():
    truth = {0: [4, 12], 1: [8]}
    X = _recording(20, truth, n_ch=2)
    _, discharges, *_ = SpikeDetectorHilbert().run(X, FS)
    for key in ('MV', 'MA', 'MP', 'MD', 'MW', 'MPDF'):
        assert discharges[key].shape[1] == 2


# =========================================================================================
# detect_spikes_janca (eeg_forge formulation)
# =========================================================================================
import os
import warnings
from pathlib import Path

from brainmaze_eeg.spikes import detect_spikes_janca, janca_resampling
from brainmaze_eeg.tests.spike_synth import synth_ieeg

_FIXTURE = Path(__file__).parent / 'data' / 'janca_eeg_forge_reference.npz'
_REF = np.load(_FIXTURE)
_CASES = sorted(k for k in _REF.files if k.startswith('fs'))


def _case(key):
    fs, seed = key[2:].split('_seed')
    return int(fs), int(seed)


@pytest.mark.parametrize('key', _CASES)
def test_janca_matches_eeg_forge_reference_fixture(key):
    """Detection indices identical to eeg_forge's spike_detection_Janca (default params)."""
    fs, seed = _case(key)
    x, _ = synth_ieeg(fs, dur=float(_REF['dur']), seed=seed)
    np.testing.assert_array_equal(detect_spikes_janca(x, fs), _REF[key])


@pytest.mark.skipif(not os.environ.get('EEG_FORGE_PATH'),
                    reason='set EEG_FORGE_PATH to an eeg_forge clone for the live comparison')
@pytest.mark.parametrize('fs', [200, 256, 512, 2048, 5000])
def test_janca_matches_live_eeg_forge(fs):
    from brainmaze_eeg.tests.data.make_janca_reference_fixtures import load_reference
    ref = load_reference(os.environ['EEG_FORGE_PATH'])
    x, _ = synth_ieeg(fs, dur=60.0, seed=7)
    np.testing.assert_array_equal(detect_spikes_janca(x, fs), ref(x, fs))


def _hits(det, truth, tol):
    return sum(bool(np.any(np.abs(det - t) <= tol)) for t in truth)


def test_janca_finds_injected_spikes():
    x, truth = synth_ieeg(500, dur=60.0, seed=3, amp_range=(250, 400))
    det = detect_spikes_janca(x, 500)
    assert _hits(det, truth, 0.05 * 500) >= 0.9 * truth.size
    assert det.size <= truth.size + 3


def test_janca_works_at_32k_where_reference_is_unstable():
    x, truth = synth_ieeg(32000, dur=30.0, seed=3, amp_range=(250, 400))
    det = detect_spikes_janca(x, 32000)
    assert _hits(det, truth, 0.05 * 32000) >= 0.9 * truth.size


@pytest.mark.parametrize('scale', [1e-6, 1e-8, 1e3])
def test_janca_is_scale_invariant(scale):
    x, _ = synth_ieeg(500, dur=60.0, seed=0)
    np.testing.assert_array_equal(detect_spikes_janca(x * scale, 500), detect_spikes_janca(x, 500))


def test_janca_multichannel_equals_per_channel():
    a, _ = synth_ieeg(512, dur=30.0, seed=0)
    b, _ = synth_ieeg(512, dur=30.0, seed=1)
    out = detect_spikes_janca(np.vstack([a, b]), 512)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_array_equal(out[0], detect_spikes_janca(a, 512))
    np.testing.assert_array_equal(out[1], detect_spikes_janca(b, 512))


def test_janca_transposed_input_raises():
    with pytest.raises(ValueError, match='transpose'):
        detect_spikes_janca(np.zeros((5000, 4)), 500)


@pytest.mark.parametrize('kw', [dict(band=(10, 150)), dict(band=(60, 10)), dict(band=(0, 60)),
                                dict(filter_order=0), dict(window_s=0), dict(threshold=-1),
                                dict(min_distance_s=-0.1), dict(target_fs=-5),
                                dict(decimation='bogus'),
                                dict(powerline=-50), dict(notch_width=0),
                                dict(notch_harmonics=0)])
def test_janca_invalid_parameters_raise(kw):
    x, _ = synth_ieeg(500, dur=10.0, seed=0)
    with pytest.raises(ValueError):
        detect_spikes_janca(x, 500, **kw)


def test_janca_notch_order_validated_even_if_all_notches_skipped():
    from brainmaze_eeg.spikes import design_janca_filters
    with pytest.raises(ValueError, match='notch_order'):
        design_janca_filters(80, band=(10, 30), notch_order=0)    # 50 Hz notch skipped at 80 Hz


def test_janca_band_above_analysis_nyquist_raises_with_hint():
    x, _ = synth_ieeg(2000, dur=10.0, seed=0, mains_hz=None)
    with pytest.raises(ValueError, match='analysis'):
        detect_spikes_janca(x, 2000, band=(80, 250))          # analysis rate 200 Hz
    # a ripple-band configuration works once the analysis rate allows it
    det = detect_spikes_janca(x, 2000, band=(80, 250), target_fs=1000)
    assert det.dtype == np.int64


def test_janca_notch_above_nyquist_is_skipped_with_warning():
    x, _ = synth_ieeg(100, dur=30.0, seed=0, mains_hz=None)
    with pytest.warns(UserWarning, match='skipped'):
        detect_spikes_janca(x, 100, band=(10, 40))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        detect_spikes_janca(x, 100, band=(10, 40), powerline=None)


@pytest.mark.parametrize('fs,expect', [(200, (1, 1, 200.0)), (399, (1, 1, 399.0)),
                                       (400, (1, 2, 200.0)), (512, (1, 2, 256.0)),
                                       (2048, (1, 10, 204.8)), (32000, (1, 160, 200.0))])
def test_janca_integer_decimation_rule(fs, expect):
    assert janca_resampling(fs, 200.0, 'integer') == expect


@pytest.mark.parametrize('fs', [256, 512, 2048, 30000])
def test_janca_exact_resampling_reaches_target(fs):
    up, down, fs_a = janca_resampling(fs, 200.0, 'exact')
    assert fs_a == pytest.approx(200.0, rel=1e-12)
    x, truth = synth_ieeg(fs, dur=30.0, seed=3, amp_range=(250, 400))
    det, info = detect_spikes_janca(x, fs, decimation='exact', return_details=True)
    assert info['fs_analysis'] == pytest.approx(200.0)
    assert _hits(det, truth, 0.05 * fs) >= 0.9 * truth.size


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_janca_raw_detector_raises_on_non_finite(bad):
    # the raw detector never silently returns zero detections (the reference's behaviour)
    x, _ = synth_ieeg(500, dur=10.0, seed=0)
    x[1000] = bad
    with pytest.raises(ValueError, match='GapAwareSpikeDetector'):
        detect_spikes_janca(x, 500)


def test_janca_return_details():
    x, _ = synth_ieeg(512, dur=20.0, seed=0)
    det, info = detect_spikes_janca(x, 512, return_details=True)
    assert info['fs_analysis'] == 256.0 and (info['up'], info['down']) == (1, 2)
    assert info['envelope'].shape == info['threshold'].shape == (x.size // 2,)
    assert np.all(info['envelope'][det // 2] > info['threshold'][det // 2])
    assert set(info) == {'fs_analysis', 'up', 'down', 'envelope', 'threshold', 'filters',
                         'preset', 'params'}
    assert info['preset'] == 'spike' and info['params']['band'] == (10.0, 60.0)


def test_v24_raw_raises_on_nan_and_detect_protocol():
    fs = 512
    x, truth = synth_ieeg(fs, dur=60.0, seed=3, amp_range=(250, 400))
    out, *_ = SpikeDetectorHilbert().run(x, fs)
    det = SpikeDetectorHilbert().detect(x[None, :], fs)
    assert len(det) == 1
    np.testing.assert_array_equal(det[0], np.unique(np.round(out['pos'] * fs).astype(int)))
    xg = x.copy()
    xg[30 * fs] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        SpikeDetectorHilbert().run(xg, fs)


def test_janca_detector_object():
    from brainmaze_eeg.spikes import JancaDetector
    x, _ = synth_ieeg(512, dur=30.0, seed=0)
    X = np.vstack([x, x[::-1]])
    a = JancaDetector(threshold=4.0).detect(X, 512)
    b = detect_spikes_janca(X, 512, threshold=4.0)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    with pytest.raises(TypeError):
        JancaDetector(nonsense=1)
    with pytest.raises(ValueError):
        JancaDetector().detect(x, 512)          # protocol is 2-D


@pytest.mark.parametrize('fs', [204.8, 1000.5])
def test_v24_non_integer_rate_resamples_exactly(fs):
    det = SpikeDetectorHilbert()
    x = np.sin(2 * np.pi * 30 * np.arange(int(20 * fs)) / fs)[:, None]
    y = det._resample(x, fs, 200.0)
    assert y.shape[0] == pytest.approx(20 * 200, abs=1)
    f = np.fft.rfftfreq(y.shape[0], 1 / 200)
    assert f[np.abs(np.fft.rfft(y[:, 0])).argmax()] == pytest.approx(30, abs=0.06)
