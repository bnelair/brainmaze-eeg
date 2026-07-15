import numpy as np
import pytest

from brainmaze_utils.signal import fft_filter
from brainmaze_eeg.features.wave_detector import (
    WaveDetector,
    detect_waves,
    _bandpass_fft,
)

FS = 200


def _sine(f, amp=1.0, dur=10.0, fs=FS):
    t = np.arange(0, dur, 1 / fs)
    return amp * np.sin(2 * np.pi * f * t)


# ----------------------------------------------------------------------------------
# one-pass band-pass (must equal the two-pass hp -> lp it replaces)
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('n', [1999, 2000, 6000])
def test_bandpass_equals_hp_then_lp(n):
    x = np.random.default_rng(0).standard_normal(n)
    ref = fft_filter(fft_filter(x, FS, 0.5, 'hp'), FS, 4.0, 'lp')
    np.testing.assert_allclose(_bandpass_fft(x, FS, 0.5, 4.0), ref, atol=1e-9)


# ----------------------------------------------------------------------------------
# #35 amplitude recovery across bands (the 'lp'->'hp' fix)
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('lo, hi, f0', [(0.5, 4, 2), (4, 8, 6), (8, 12, 10), (11, 16, 13)])
def test_amplitude_recovered_in_every_band(lo, hi, f0):
    # a unit sine has peak-to-peak 2.0; the old 'lp' bug collapsed this to ~0
    d = detect_waves(_sine(f0, amp=1.0), FS, fband=(lo, hi))
    assert d['min_pos'].size >= int(10 * f0) - 3
    assert np.nanmean(d['pk2pk']) == pytest.approx(2.0, abs=0.1)
    # min is a negative trough, max a positive peak
    assert np.nanmean(d['min_val']) < 0 < np.nanmean(d['max_val'])


def test_delta_t_matches_half_period():
    d = detect_waves(_sine(2.0), FS, fband=(0.5, 4))
    assert np.nanmean(d['delta_t']) == pytest.approx(1 / (2 * 2.0), abs=1e-3)


# ----------------------------------------------------------------------------------
# #47 slope conventions
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('f, amp', [(0.75, 1.0), (0.75, 50.0), (2.0, 1.0)])
def test_downslope_matches_analytic(f, amp):
    # for A*sin(2*pi*f*t): trough is a quarter period after the down zero-crossing,
    # so downslope = |A| / (T/4) = 4*A*f
    band = (0.5, 0.9) if f < 1 else (1.0, 3.9)
    d = detect_waves(_sine(f, amp=amp, dur=20, fs=500), 500, fband=band)
    assert np.nanmean(d['downslope']) == pytest.approx(4 * amp * f, rel=0.05)


def test_upslope_matches_analytic():
    # upslope = pk2pk / (T/2) = 2A / (1/(2f)) = 4*A*f  (same for a symmetric sine)
    d = detect_waves(_sine(2.0, amp=1.0, dur=20, fs=500), 500, fband=(1.0, 3.9))
    assert np.nanmean(d['upslope']) == pytest.approx(4 * 1.0 * 2.0, rel=0.05)


def test_slope_selection_switches_reported_feature():
    x = _sine(2.0, amp=1.0, dur=30, fs=500)
    down = WaveDetector(fs=500, fband=(1, 3.9), slope='downslope')(x)
    up = WaveDetector(fs=500, fband=(1, 3.9), slope='upslope')(x)
    names = down[1]
    i = names.index('WAVE_SLOPE_MEAN')
    # both ~4*A*f here, but they come from different code paths; check they are finite
    assert np.isfinite(np.nanmean(down[0][i]))
    assert np.isfinite(np.nanmean(up[0][i]))


def test_measure_on_reads_amplitude_from_the_supplied_trace():
    t = np.arange(0, 20, 1 / 500)
    narrow = 50 * np.sin(2 * np.pi * 0.75 * t)
    broad = narrow + 8 * np.sin(2 * np.pi * 20 * t)   # fast content only in the broadband trace
    base = detect_waves(narrow, 500, fband=(0.5, 0.9))
    on = detect_waves(narrow, 500, fband=(0.5, 0.9), measure_on=broad)
    # detection positions identical, but measured amplitudes differ (broadband is noisier)
    np.testing.assert_array_equal(base['min_pos'], on['min_pos'])
    assert not np.allclose(base['min_val'], on['min_val'])


# ----------------------------------------------------------------------------------
# amplitude threshold
# ----------------------------------------------------------------------------------
def test_amplitude_threshold_drops_shallow_waves():
    t = np.arange(0, 20, 1 / 500)
    x = 50 * np.sin(2 * np.pi * 0.75 * t) * (t < 10) + 2 * np.sin(2 * np.pi * 0.75 * t) * (t >= 10)
    n_all = WaveDetector(fs=500, fband=(0.5, 0.9)).detect(x)['min_pos'].size
    n_thr = WaveDetector(fs=500, fband=(0.5, 0.9), amplitude_threshold=5).detect(x)['min_pos'].size
    assert n_thr < n_all
    kept = WaveDetector(fs=500, fband=(0.5, 0.9), amplitude_threshold=5).detect(x)
    assert np.all(-kept['min_val'] >= 5)


# ----------------------------------------------------------------------------------
# #42 interface: shapes, per-signal independence, empty windows
# ----------------------------------------------------------------------------------
def test_call_returns_values_names_1d():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5)
    values, names = det(_sine(2.0, dur=30))
    assert names[0] == 'WAVE_RATE'
    assert all(v.shape == (6,) for v in values)


def test_call_2d_shape_and_datarate():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5, datarate=True)
    X = np.vstack([_sine(2.0, dur=30), _sine(2.0, dur=30)])
    values, names = det(X)
    assert names[0] == 'DATA_RATE'
    assert all(v.shape == (2, 6) for v in values)


def test_signals_are_detected_independently():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5)
    sig = _sine(2.0, amp=5, dur=30)
    stacked, names = det(np.vstack([sig, np.zeros_like(sig)]))
    alone, _ = det(sig)
    ri = names.index('WAVE_RATE')
    np.testing.assert_allclose(stacked[0][ri], alone[ri], equal_nan=True)   # row 0 unchanged by row 1
    assert np.all(stacked[ri][1] == 0)                                       # zero channel -> no waves


def test_empty_windows_give_zero_rate_and_nan_shape():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5, datarate=True)
    values, names = det(np.zeros(30 * FS))
    got = dict(zip(names, values))
    np.testing.assert_array_equal(got['WAVE_RATE'], np.zeros(6))
    assert np.all(np.isnan(got['WAVE_PK2PK_MEAN']))
    np.testing.assert_array_equal(got['DATA_RATE'], np.ones(6))


def test_flat_and_short_signals_do_not_raise():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert det.detect(np.zeros(2000))['min_pos'].size == 0
    assert det.detect(np.zeros(2))['min_pos'].size == 0     # shorter than 3 samples


def test_1d_detect_returns_dict_list_returns_list():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert isinstance(det.detect(_sine(2.0)), dict)
    out = det.detect([_sine(2.0), _sine(2.0)])
    assert isinstance(out, list) and len(out) == 2


def test_whole_signal_is_one_window_when_segm_size_none():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    values, names = det(_sine(2.0, dur=10))
    # 1-D input -> (n_windows,) per feature; segm_size=None means a single window
    assert all(v.shape == (1,) for v in values)
    assert values[names.index('WAVE_RATE')][0] > 0


# ----------------------------------------------------------------------------------
# construction validation
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs', [
    {'fs': 0},
    {'fs': FS, 'fband': (4, 1)},          # low >= high
    {'fs': FS, 'fband': (0.5, 200)},      # above Nyquist
    {'fs': FS, 'slope': 'sideways'},
    {'fs': FS, 'segm_size': 5, 'overlap': 5},
    {'fs': FS, 'n_processes': 0},
])
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        WaveDetector(**kwargs)


def test_backward_compatible_cutoff_aliases():
    det = WaveDetector(fs=FS, cutoff_low=0.5, cutoff_high=4)
    assert det.fband == (0.5, 4.0)
    assert det.cutoff_low == 0.5 and det.cutoff_high == 4.0


# ----------------------------------------------------------------------------------
# multiprocessing and measure_on plumbing
# ----------------------------------------------------------------------------------
def test_n_processes_matches_serial():
    sigs = [_sine(2.0, amp=5, dur=20), _sine(3.0, amp=5, dur=20)]
    serial = WaveDetector(fs=FS, fband=(0.5, 4), n_processes=1).detect(sigs)
    parallel = WaveDetector(fs=FS, fband=(0.5, 4), n_processes=2).detect(sigs)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a['min_pos'], b['min_pos'])


def test_measure_on_2d_and_length_check():
    X = np.vstack([_sine(0.75, amp=50, dur=20, fs=500), _sine(0.75, amp=50, dur=20, fs=500)])
    det = WaveDetector(fs=500, fband=(0.5, 0.9), segm_size=10)
    values, names = det(X, measure_on=X)          # 2-D measure_on accepted
    assert all(v.shape == (2, 2) for v in values)
    with pytest.raises(ValueError):
        det(X, measure_on=X[0])                    # mismatched signal count
