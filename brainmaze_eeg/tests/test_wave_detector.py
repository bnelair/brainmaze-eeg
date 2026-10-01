import time
import warnings

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

from brainmaze_eeg.features.wave_detector import (
    WaveDetector,
    detect_waves,
    _argext_groups,
    _bandpass_fft,
    _excluded_mask,
    _filter_signals,
)

FS = 200


def _sine(f, amp=1.0, dur=10.0, fs=FS, phase=0.0):
    t = np.arange(0, dur, 1 / fs)
    return amp * np.sin(2 * np.pi * f * t + phase)


def _pink(n, fs, rms=30.0, seed=0):
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    scale = np.zeros_like(freqs)
    scale[1:] = 1 / np.sqrt(freqs[1:])
    x = np.fft.irfft((rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)) * scale, n)
    return x * rms / x.std()


def _feats(out):
    values, names = out
    return dict(zip(names, values))


def _sine_wave_spans(f, phase, dur, fs):
    """Analytic (zero_crossing_time, end_time) of every complete wave of sin(2 pi f t + phase)."""
    # down-going crossings: 2 pi f t + phase = pi + 2 pi k
    k = np.arange(-2, int(dur * f) + 3)
    t0 = (np.pi + 2 * np.pi * k - phase) / (2 * np.pi * f)
    t1 = t0 + 1 / f                                   # next down-going crossing
    keep = (t0 > 0) & (t1 < (np.ceil(dur * fs) - 1) / fs)
    return t0[keep], t1[keep]


# ----------------------------------------------------------------------------------
# filters: measured response == design
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('fs, band', [(200, (0.5, 4.0)), (200, (0.5, 0.9)), (200, (4.0, 8.0)),
                                      (1000, (0.5, 4.0)), (5000, (0.5, 0.9)), (25000, (0.5, 4.0))])
def test_butterworth_measured_response_matches_design(fs, band):
    # (the same check was run offline for every fs x band combination up to 25 kHz)
    lo, hi = band
    sos = butter(2, [lo, hi], btype='bandpass', fs=fs, output='sos')
    for f in (lo / 4, lo, np.sqrt(lo * hi), hi, 4 * hi):
        dur = max(20.0, 20.0 / f)
        t = np.arange(int(dur * fs)) / fs
        x = np.sin(2 * np.pi * f * t)
        xn, _, _ = _filter_signals(x - x.mean(), float(fs), lo, hi, 'butter', 2, 0)
        m = slice(t.size // 4, -t.size // 4)
        measured = np.sqrt(2) * xn[m].std()
        design = abs(sosfreqz(sos, worN=[f], fs=fs)[1][0]) ** 2     # forward-backward
        assert measured == pytest.approx(design, abs=5e-3)
        if f in (lo, hi):
            assert measured == pytest.approx(0.5, abs=5e-3)            # -6 dB at the band edges
        # zero phase: the output is the input scaled by the gain, with no lag
        np.testing.assert_allclose(xn[m], measured * x[m], atol=5e-3)


@pytest.mark.parametrize('fs', [200, 5000])
def test_drift_highpass_preserves_band_and_removes_drift(fs):
    t = np.arange(int(80 * fs)) / fs
    m = slice(t.size // 4, -t.size // 4)
    for f, expect in ((0.5, 0.996), (0.125, 0.004)):
        _, xhp, _ = _filter_signals(np.sin(2 * np.pi * f * t), float(fs), 0.5, 4.0, 'butter', 2, 0)
        assert np.sqrt(2) * xhp[m].std() == pytest.approx(expect, abs=2e-3)


@pytest.mark.parametrize('n', [1999, 2000, 6000])
def test_fft_bandpass_passes_inband_and_removes_outofband(n):
    t = np.arange(n) / FS
    inband = np.sin(2 * np.pi * 2.0 * t)
    below = np.sin(2 * np.pi * 0.2 * t)
    above = np.sin(2 * np.pi * 10.0 * t)
    m = slice(n // 5, -n // 5)   # ignore FFT edge ripple of the ideal filter
    np.testing.assert_allclose(_bandpass_fft(inband, FS, 0.5, 4.0)[m], inband[m], atol=1e-2)
    assert np.abs(_bandpass_fft(below, FS, 0.5, 4.0)[m]).max() < 1e-2
    assert np.abs(_bandpass_fft(above, FS, 0.5, 4.0)[m]).max() < 1e-2


def test_isolated_transient_gives_one_wave_not_ringing():
    # one 50 uV, 1 Hz cycle in 30 s of zeros: the Butterworth path finds that one wave;
    # the brick-wall FFT filter rings and invents side-lobe "waves" around it
    x = np.zeros(30 * FS)
    x[15 * FS:16 * FS] = -50 * np.sin(2 * np.pi * np.arange(FS) / FS)
    butter_det = detect_waves(x, FS, (0.5, 4))
    fft_det = detect_waves(x, FS, (0.5, 4), filter='fft')
    assert butter_det['min_pos'].size == 1
    assert abs(butter_det['min_pos'][0] - (15 * FS + FS // 4)) <= 3
    assert fft_det['min_pos'].size > 3


# ----------------------------------------------------------------------------------
# counting: exact counts, edges, band edge
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('f, band', [(2.0, (0.5, 4)), (1.0, (0.5, 4)), (6.0, (4, 8)),
                                     (3.9, (1, 3.9)), (1.0, (1, 3.9)), (0.75, (0.5, 0.9))])
def test_exact_counts_with_random_phases(f, band):
    # every analytic wave whose span lies inside the analysable part is detected once,
    # and nothing else; a 2-sample guard around the margin borders avoids ties
    rng = np.random.default_rng(42)
    dur = 60.0
    m = 3 / band[0]
    for _ in range(10):
        phase = rng.uniform(0, 2 * np.pi)
        d = detect_waves(_sine(f, dur=dur, phase=phase), FS, band)
        t0, t1 = _sine_wave_spans(f, phase, dur, FS)
        n_end = (np.ceil(dur * FS) - 1) / FS
        inside = (t0 >= m) & (t1 <= n_end - m)
        border = (np.abs(t0 - m) < 2 / FS) | (np.abs(t1 - (n_end - m)) < 2 / FS)
        if border.any():
            continue
        assert d['min_pos'].size == int(inside.sum()), (f, band, phase)
        # each detection sits on its analytic wave
        np.testing.assert_allclose(d['zero_pos_frac'] / FS, t0[inside], atol=1.5 / FS)


def test_truncated_edge_half_waves_are_never_paired():
    # sin(2 pi 2 t - pi/4) starts inside a positive half-wave falling to a truncated
    # trough; v1.0.0 paired it (pk2pk 1.707). With margins 0 the first wave is complete.
    x = _sine(2.0, dur=30, phase=-np.pi / 4)
    d = detect_waves(x, FS, (0.5, 4), edge_margin_s=0)
    t0, _ = _sine_wave_spans(2.0, -np.pi / 4, 30, FS)
    # first detection = first complete wave (its crossing is shifted a few samples by
    # the filter's edge transient, which is what edge_margin_s guards against)
    assert d['zero_pos_frac'][0] / FS == pytest.approx(t0[0], abs=0.03)
    assert d['zero_pos'][0] > 0
    # amplitudes are distorted near the edge without a margin; interior ones are exact
    interior = (d['zero_pos_frac'] / FS > 6) & (d['end_pos'] / FS < 24)
    np.testing.assert_allclose(d['pk2pk'][interior], 2.0, atol=0.02)


def test_edge_margin_excludes_edge_waves_and_rate_stays_unbiased():
    x = _sine(2.0, dur=60)
    d0 = detect_waves(x, FS, (0.5, 4), edge_margin_s=0)
    d = detect_waves(x, FS, (0.5, 4))                       # default 3 / 0.5 = 6 s
    assert d['min_pos'].size < d0['min_pos'].size
    assert d['zero_pos_frac'].min() / FS >= 6.0
    assert d['end_pos'].max() / FS <= (x.size - 1) / FS - 6.0
    rate = WaveDetector(fs=FS, fband=(0.5, 4))(x)[0][0]      # one window
    assert rate[0] == pytest.approx(2.0, abs=2 / 48)        # +/- one wave per border, 48 s


def test_upper_band_edge_frequency_is_detected():
    # v1.0.0's strict duration gate lost ~1/3 of 3.9 Hz waves in band (1, 3.9)
    # (+/- one wave lost at each of the two excluded-zone borders)
    rate = WaveDetector(fs=FS, fband=(1, 3.9))(_sine(3.9, dur=60))[0][0][0]
    assert rate == pytest.approx(3.9, abs=2 / 54)


# ----------------------------------------------------------------------------------
# amplitudes, slopes, positions
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('lo, hi, f0', [(0.5, 4, 2), (4, 8, 6), (8, 12, 10), (11, 16, 13)])
def test_amplitude_recovered_in_every_band(lo, hi, f0):
    # a unit sine has peak-to-peak 2.0 on the unfiltered signal; the filtered (band)
    # amplitude is scaled by the filter gain |H(f0)|^2
    dur = 30
    d = detect_waves(_sine(f0, amp=1.0, dur=dur), FS, fband=(lo, hi))
    assert d['min_pos'].size >= int((dur - 2 * 3 / lo) * f0) - 2
    assert np.mean(d['pk2pk']) == pytest.approx(2.0, abs=0.02)
    assert np.mean(d['min_val']) < 0 < np.mean(d['max_val'])
    sos = butter(2, [lo, hi], btype='bandpass', fs=FS, output='sos')
    gain = abs(sosfreqz(sos, worN=[f0], fs=FS)[1][0]) ** 2
    assert np.mean(d['pk2pk_band']) == pytest.approx(2.0 * gain, rel=0.02)


def test_delta_t_matches_half_period():
    d = detect_waves(_sine(2.0, dur=30), FS, fband=(0.5, 4))
    assert np.mean(d['delta_t']) == pytest.approx(1 / (2 * 2.0), abs=1e-3)


@pytest.mark.parametrize('f, amp, fs', [(0.75, 1.0, 500), (0.75, 50.0, 500), (2.0, 1.0, 500),
                                        (2.0, 50.0, 200), (1.0, 50.0, 200)])
def test_downslope_matches_analytic(f, amp, fs):
    # A*sin(2 pi f t): trough a quarter period after the down crossing -> 4*A*f.
    # The interpolated zero crossing keeps this within 1 % at fs=200 (v1.0.0: +2-4 %).
    band = (0.5, 0.9) if f < 1 else (0.5, 4) if f == 1.0 else (1.0, 3.9)
    d = detect_waves(_sine(f, amp=amp, dur=60, fs=fs, phase=0.3), fs, fband=band)
    assert np.mean(d['downslope']) == pytest.approx(4 * amp * f, rel=0.01)
    assert np.mean(d['downslope_band']) > 0


def test_upslope_matches_analytic():
    d = detect_waves(_sine(2.0, amp=1.0, dur=20, fs=500), 500, fband=(1.0, 3.9))
    assert np.mean(d['upslope']) == pytest.approx(4 * 1.0 * 2.0, rel=0.02)


@pytest.mark.parametrize('refine', [0, 2.0, 4.0, None])
@pytest.mark.parametrize('band', [(0.5, 4), (0.5, 0.9), (4, 8)])
def test_positions_are_ordered_and_inside_their_half_waves(refine, band):
    x = _pink(120 * FS, FS, seed=3)
    d = detect_waves(x, FS, band, refine_lowpass=refine)
    assert d['min_pos'].size > 10
    for sfx in ('', '_band'):
        assert np.all(d['zero_pos_frac'] < d['min_pos' + sfx])
        assert np.all(d['min_pos' + sfx] < d['max_pos' + sfx])
        assert np.all(d['max_pos' + sfx] <= d['end_pos'])
        assert np.all(d['down_dur' + sfx] > 0)
        assert np.all(np.isfinite(d['downslope' + sfx]))
    # troughs are on the negative half-wave of the filtered signal, peaks on the positive one
    assert np.all(d['min_val_band'] < 0) and np.all(d['max_val_band'] > 0)
    assert np.all(np.diff(d['min_pos']) > 0)
    if refine == 0:
        np.testing.assert_array_equal(d['min_pos'], d['min_pos_band'])


def test_noise_bias_of_amplitudes_by_refinement():
    # 100 uV pk2pk 1 Hz sine + white noise SD 10 uV: the default (positions from the
    # filtered signal) is unbiased; refining on the broadband signal picks noise extremes
    rng = np.random.default_rng(0)
    x = _sine(1.0, amp=50, dur=120) + 10 * rng.standard_normal(120 * FS)
    p0 = np.mean(detect_waves(x, FS, (0.5, 4))['pk2pk'])
    p4 = np.mean(detect_waves(x, FS, (0.5, 4), refine_lowpass=4)['pk2pk'])
    pn = np.mean(detect_waves(x, FS, (0.5, 4), refine_lowpass=None)['pk2pk'])
    assert p0 == pytest.approx(100, rel=0.03)
    assert p0 < p4 < pn
    assert pn > 125


def test_measure_on_reads_amplitude_from_the_supplied_trace():
    t = np.arange(0, 60, 1 / 500)
    narrow = 50 * np.sin(2 * np.pi * 0.75 * t)
    broad = narrow + 8 * np.sin(2 * np.pi * 20 * t)
    base = detect_waves(narrow, 500, fband=(0.5, 0.9))
    on = detect_waves(narrow, 500, fband=(0.5, 0.9), measure_on=broad)
    np.testing.assert_array_equal(base['min_pos'], on['min_pos'])
    np.testing.assert_allclose(on['min_val'], broad[on['min_pos']] - broad.mean())
    np.testing.assert_allclose(base['min_val_band'], on['min_val_band'])


# ----------------------------------------------------------------------------------
# gaps
# ----------------------------------------------------------------------------------
def test_single_nan_does_not_kill_the_record():
    # v1.0.0: one NaN -> WAVE_RATE 0 everywhere while DATA_RATE ~1
    x = _sine(2.0, amp=20, dur=120)
    x[50 * FS] = np.nan
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30, datarate=True)
    values, names = det(x)
    got = dict(zip(names, values))
    np.testing.assert_allclose(got['DATA_RATE'], [1, 1 - 1 / (30 * FS), 1, 1])
    # every window's rate is the true 2 Hz up to one wave per analysable time
    ana = 30 - np.array([6, 2 * 6 + 1 / FS, 0, 6])
    assert np.all(np.abs(got['WAVE_RATE'] - 2.0) <= 1 / ana + 1e-9)
    d = det.detect(x)
    assert d['gaps'].tolist() == [[50 * FS, 50 * FS + 1]]
    # no wave span touches the NaN +/- margin
    assert not np.any((d['zero_pos_frac'] / FS < 50 + 6 + 1 / FS) & (d['end_pos'] / FS >= 50 - 6))


def test_gaps_rate_is_normalised_by_analysable_time():
    rng = np.random.default_rng(5)
    x = _sine(1.5, amp=40, dur=600, phase=0.2)
    for _ in range(12):
        L = int(rng.uniform(0.05, 4) * FS)
        s = int(rng.integers(0, x.size - L))
        x[s:s + L] = np.nan
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=60, datarate=True)
    got = _feats(det(x))
    d = det.detect(x)
    excl = _excluded_mask(x.size, d['gaps'], FS, det.gap_margin_s, det.edge_margin_s)
    ana = (~excl & np.isfinite(x)).reshape(10, -1).sum(axis=1) / FS
    ok = ana > 10
    # rate == 1.5 Hz up to a few waves lost at excluded-zone borders
    assert np.all(np.abs(got['WAVE_RATE'][ok] - 1.5) <= 3 / ana[ok])
    np.testing.assert_allclose(got['DATA_RATE'], np.isfinite(x).reshape(10, -1).mean(axis=1))
    # the wave count used per window equals the detections whose trough is in it
    n_in = np.histogram(d['min_pos'], bins=np.arange(11) * 60 * FS)[0]
    np.testing.assert_allclose(got['WAVE_RATE'][ok], (n_in / ana)[ok])


def test_gap_at_start_and_end_and_all_nan():
    x = _sine(2.0, dur=60)
    x[:3 * FS] = np.nan
    x[-2 * FS:] = np.nan
    d = detect_waves(x, FS, (0.5, 4))
    assert d['min_pos'].size > 0
    assert d['zero_pos_frac'].min() / FS >= 3 + 6
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=10, datarate=True)
    got = _feats(det(np.full(30 * FS, np.nan)))
    np.testing.assert_array_equal(got['DATA_RATE'], np.zeros(3))
    assert np.all(np.isnan(got['WAVE_RATE']))


def test_nan_in_measure_on_is_a_gap_and_inf_too():
    x = _sine(2.0, amp=20, dur=60)
    m = x.copy()
    m[30 * FS:31 * FS] = np.nan
    x2 = x.copy()
    x2[10 * FS] = np.inf
    assert detect_waves(x, FS, (0.5, 4), measure_on=m)['gaps'].tolist() == [[30 * FS, 31 * FS]]
    assert detect_waves(x2, FS, (0.5, 4))['gaps'].tolist() == [[10 * FS, 10 * FS + 1]]


def test_nan_policy_raise():
    x = _sine(2.0, dur=20)
    x[5] = np.nan
    with pytest.raises(ValueError, match='nan_policy'):
        detect_waves(x, FS, (0.5, 4), nan_policy='raise')
    with pytest.raises(ValueError):
        WaveDetector(fs=FS, nan_policy='raise')(x)


def test_return_signals_masks_gaps():
    x = _sine(2.0, dur=30)
    x[1000:1100] = np.nan
    d = WaveDetector(fs=FS, fband=(0.5, 4)).detect(x, return_signals=True)
    for k in ('x_band', 'x_amp'):
        assert d[k].shape == x.shape
        assert np.all(np.isnan(d[k][1000:1100]))
        assert np.all(np.isfinite(d[k][:1000]))
    np.testing.assert_allclose(d['x_band'][d['min_pos_band']], d['min_val_band'])


# ----------------------------------------------------------------------------------
# interface: shapes, names, per-signal independence, empty windows
# ----------------------------------------------------------------------------------
def test_call_returns_values_names_1d():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5)
    values, names = det(_sine(2.0, dur=30))
    assert names == det.feature_names
    assert names[0] == 'WAVE_RATE'
    assert all(v.shape == (6,) for v in values)


def test_call_2d_shape_and_datarate():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=5, datarate=True)
    X = np.vstack([_sine(2.0, dur=30), _sine(2.0, dur=30)])
    values, names = det(X)
    assert names[0] == 'DATA_RATE'
    assert all(v.shape == (2, 6) for v in values)
    lst, _ = det([X[0], X[1]])
    for a, b in zip(values, lst):
        np.testing.assert_array_equal(a, b)


def test_features_on_band_and_both():
    x = _sine(2.0, amp=10, dur=60)
    bb = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30)
    band = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30, features_on='band')
    both = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30, features_on='both')
    vb, nb = bb(x)
    vn, nn = band(x)
    vo, no = both(x)
    assert nb == ['WAVE_RATE', 'WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN', 'WAVE_DELTA_T_MEAN',
                  'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN']
    assert nn == ['WAVE_RATE'] + [k + '_BAND' for k in nb[1:]]
    assert no == nb + nn[1:]
    got = dict(zip(no, vo))
    for k, v in zip(nb, vb):
        np.testing.assert_array_equal(got[k], v)
    for k, v in zip(nn, vn):
        np.testing.assert_array_equal(got[k], v)
    sos = butter(2, [0.5, 4], btype='bandpass', fs=FS, output='sos')
    gain = abs(sosfreqz(sos, worN=[2.0], fs=FS)[1][0]) ** 2
    np.testing.assert_allclose(got['WAVE_PK2PK_MEAN'], 20, rtol=0.01)
    np.testing.assert_allclose(got['WAVE_PK2PK_MEAN_BAND'], 20 * gain, rtol=0.01)


def test_windows_match_bruteforce_means():
    x = _pink(300 * FS, FS, seed=9)
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=20, overlap=5, features_on='both',
                       slope='upslope')
    got = _feats(det(x))
    d = det.detect(x)
    starts = np.arange(0, x.size - 20 * FS + 1, 15 * FS)
    for i, s in enumerate(starts):
        sel = (d['min_pos'] >= s) & (d['min_pos'] < s + 20 * FS)
        if sel.any():
            assert got['WAVE_PK2PK_MEAN'][i] == pytest.approx(d['pk2pk'][sel].mean())
            assert got['WAVE_SLOPE_MEAN_BAND'][i] == pytest.approx(d['upslope_band'][sel].mean())
        else:
            assert np.isnan(got['WAVE_PK2PK_MEAN'][i])


def test_signals_are_detected_independently():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=10)
    sig = _sine(2.0, amp=5, dur=60)
    stacked, names = det(np.vstack([sig, np.zeros_like(sig)]))
    alone, _ = det(sig)
    ri = names.index('WAVE_RATE')
    np.testing.assert_allclose(stacked[ri][0], alone[ri], equal_nan=True)
    assert np.all(stacked[ri][1] == 0)


def test_empty_windows_give_zero_rate_and_nan_shape():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=10, datarate=True)
    got = _feats(det(np.zeros(60 * FS)))
    np.testing.assert_array_equal(got['WAVE_RATE'], np.zeros(6))
    assert np.all(np.isnan(got['WAVE_PK2PK_MEAN']))
    np.testing.assert_array_equal(got['DATA_RATE'], np.ones(6))


def test_window_inside_edge_margin_has_nan_rate():
    # 2 s windows, 6 s edge margin: the first three windows have no analysable time
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=2)
    rate = det(_sine(2.0, dur=30))[0][0]
    assert np.all(np.isnan(rate[:3])) and np.all(np.isnan(rate[-3:]))
    assert np.all(np.isfinite(rate[3:-3]))


def test_flat_and_short_signals_do_not_raise():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert det.detect(np.zeros(2000))['min_pos'].size == 0
    assert det.detect(np.zeros(2))['min_pos'].size == 0
    assert det.detect(_sine(2.0, dur=1))['min_pos'].size == 0   # all inside the margins
    values, _ = WaveDetector(fs=FS, segm_size=30)(np.zeros(10 * FS))   # shorter than a window
    assert all(v.shape == (0,) for v in values)


def test_1d_detect_returns_dict_list_returns_list():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert isinstance(det.detect(_sine(2.0, dur=30)), dict)
    out = det.detect([_sine(2.0, dur=30), _sine(2.0, dur=30)])
    assert isinstance(out, list) and len(out) == 2


def test_whole_signal_is_one_window_when_segm_size_none():
    values, names = WaveDetector(fs=FS, fband=(0.5, 4))(_sine(2.0, dur=30))
    assert all(v.shape == (1,) for v in values)
    assert values[names.index('WAVE_RATE')][0] > 0


def test_transposed_input_warns():
    with pytest.warns(UserWarning, match='n_signals, n_samples'):
        WaveDetector(fs=FS).detect(np.zeros((500, 3)))


def test_numpy_scalar_fs_accepted():
    x = _sine(2.0, dur=30)
    ref = detect_waves(x, 200.0, (0.5, 4))
    for fs in (np.float32(200), np.int64(200), np.float64(200)):
        d = WaveDetector(fs=fs, fband=(0.5, 4)).detect(x)
        np.testing.assert_array_equal(d['min_pos'], ref['min_pos'])


# ----------------------------------------------------------------------------------
# slope selection, threshold
# ----------------------------------------------------------------------------------
def test_slope_selection_switches_reported_feature():
    x = _sine(2.0, amp=1.0, dur=30, fs=500, phase=0.4)
    down = _feats(WaveDetector(fs=500, fband=(1, 3.9), slope='downslope')(x))
    up = _feats(WaveDetector(fs=500, fband=(1, 3.9), slope='upslope')(x))
    d = detect_waves(x, 500, (1, 3.9))
    assert down['WAVE_SLOPE_MEAN'][0] == pytest.approx(d['downslope'].mean())
    assert up['WAVE_SLOPE_MEAN'][0] == pytest.approx(d['upslope'].mean())


def test_amplitude_threshold_drops_shallow_waves():
    t = np.arange(0, 60, 1 / 500)
    x = 50 * np.sin(2 * np.pi * 0.75 * t) * (t < 30) + 2 * np.sin(2 * np.pi * 0.75 * t) * (t >= 30)
    n_all = WaveDetector(fs=500, fband=(0.5, 0.9)).detect(x)['min_pos'].size
    kept = WaveDetector(fs=500, fband=(0.5, 0.9), amplitude_threshold=5).detect(x)
    assert kept['min_pos'].size < n_all
    assert np.all(-kept['min_val'] >= 5)
    assert kept['min_pos_band'].size == kept['min_pos'].size == kept['end_pos'].size


# ----------------------------------------------------------------------------------
# construction validation
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs', [
    {'fs': 0},
    {'fs': True},
    {'fs': FS, 'fband': (4, 1)},
    {'fs': FS, 'fband': (0, 4)},
    {'fs': FS, 'fband': (0.5, 200)},
    {'fs': FS, 'fband': (0.5, np.nan)},
    {'fs': FS, 'slope': 'sideways'},
    {'fs': FS, 'segm_size': 5, 'overlap': 5},
    {'fs': FS, 'n_processes': 0},
    {'fs': FS, 'filter': 'cheby'},
    {'fs': FS, 'filter_order': 0},
    {'fs': FS, 'gap_margin_s': -1},
    {'fs': FS, 'edge_margin_s': np.inf},
    {'fs': FS, 'refine_lowpass': 0.5},
    {'fs': FS, 'features_on': 'filtered'},
    {'fs': FS, 'amplitude_threshold': np.nan},
])
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        WaveDetector(**kwargs)


def test_backward_compatible_cutoff_aliases():
    det = WaveDetector(fs=FS, cutoff_low=0.5, cutoff_high=4)
    assert det.fband == (0.5, 4.0)
    assert det.cutoff_low == 0.5 and det.cutoff_high == 4.0


def test_default_margins_resolved():
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert det.gap_margin_s == det.edge_margin_s == 6.0
    assert WaveDetector(fs=FS, fband=(4, 8), gap_margin_s=0.1).gap_margin_s == 0.1


# ----------------------------------------------------------------------------------
# internals, multiprocessing, measure_on plumbing, speed
# ----------------------------------------------------------------------------------
def test_argext_groups_matches_numpy_including_ties():
    rng = np.random.default_rng(1)
    v = rng.integers(0, 5, 1000).astype(float)            # many ties
    starts = np.unique(np.concatenate(([0], rng.integers(1, 1000, 80))))
    bounds = np.append(starts, v.size)
    for want, f in (('min', np.argmin), ('max', np.argmax)):
        exp = [s + f(v[s:e]) for s, e in zip(bounds[:-1], bounds[1:])]
        np.testing.assert_array_equal(_argext_groups(v, starts, want), exp)


def test_n_processes_matches_serial():
    sigs = [_sine(2.0, amp=5, dur=30), _sine(3.0, amp=5, dur=30)]
    serial = WaveDetector(fs=FS, fband=(0.5, 4), n_processes=1).detect(sigs)
    parallel = WaveDetector(fs=FS, fband=(0.5, 4), n_processes=2).detect(sigs)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a['min_pos'], b['min_pos'])


def test_measure_on_2d_and_length_check():
    X = np.vstack([_sine(0.75, amp=50, dur=40, fs=500), _sine(0.75, amp=50, dur=40, fs=500)])
    det = WaveDetector(fs=500, fband=(0.5, 0.9), segm_size=20)
    values, names = det(X, measure_on=X)
    assert all(v.shape == (2, 2) for v in values)
    with pytest.raises(ValueError):
        det(X, measure_on=X[0])


def test_measure_on_length_mismatch_raises():
    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(ValueError):
        detect_waves(x, fs=FS, fband=(0.5, 4.0), measure_on=x[:-5])


def test_fft_filter_still_selectable():
    d = detect_waves(_sine(2.0, dur=60), FS, (0.5, 4), filter='fft')
    assert np.mean(d['pk2pk']) == pytest.approx(2.0, abs=0.02)


def test_speed_30min_channel():
    # generous regression bound: measured ~0.06 s per 30-min channel at 250 Hz
    fs = 250
    x = _pink(1800 * fs, fs, seed=0) + _sine(1.5, amp=60, dur=1800, fs=fs)
    det = WaveDetector(fs=fs, fband=(0.5, 4), segm_size=30, features_on='both')
    t0 = time.perf_counter()
    values, _ = det(x)
    assert time.perf_counter() - t0 < 5.0
    assert values[0].shape == (60,)
