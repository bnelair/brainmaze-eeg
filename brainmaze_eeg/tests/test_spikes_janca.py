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
