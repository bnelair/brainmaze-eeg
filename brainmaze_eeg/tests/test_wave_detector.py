import time
import warnings

import numpy as np
import pytest
from scipy.signal import butter, sosfreqz

import brainmaze_eeg.features.wave_detector as wd
from brainmaze_eeg.features.wave_detector import (
    WaveDetector,
    detect_waves,
    _argext_groups,
    _band_signal,
    _bandpass_fft,
    _default_gap_margins,
    _drift_removed,
    _paper_trace,
    _window_nanmedian,
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
        xn = _band_signal(x - x.mean(), float(fs), lo, hi, 'butter', 2)
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
        xhp = _drift_removed(np.sin(2 * np.pi * f * t), float(fs), 0.5, 'butter')
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
    m = 4 / band[0]
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
    d = detect_waves(x, FS, (0.5, 4))                       # default 4 / 0.5 = 8 s
    assert d['min_pos'].size < d0['min_pos'].size
    assert d['zero_pos_frac'].min() / FS >= 8.0
    assert d['end_pos'].max() / FS <= (x.size - 1) / FS - 8.0
    rate = WaveDetector(fs=FS, fband=(0.5, 4))(x)[0][0]      # one window
    # the denominator is shortened like the inclusion rule (R5): no border bias left
    assert rate[0] == pytest.approx(2.0, abs=1 / 48)    # periodic train: +/- 1 wave


def test_upper_band_edge_frequency_is_detected():
    # v1.0.0's strict duration gate lost ~1/3 of 3.9 Hz waves in band (1, 3.9)
    # (+/- one wave lost at each of the two excluded-zone borders)
    rate = WaveDetector(fs=FS, fband=(1, 3.9))(_sine(3.9, dur=60))[0][0][0]
    assert rate == pytest.approx(3.9, abs=1 / 54)       # periodic train: +/- 1 wave


@pytest.mark.parametrize('trough', ['refine', 'band'])
def test_duration_gate_does_not_depend_on_fs(trough):
    # R8: sub-sample durations + a relative tolerance -> the same effective band at every
    # fs (the old one-sample tolerance kept 4.05 Hz waves in (1, 3.9) at 128-200 Hz only)
    for fs in (128, 200, 500, 5000):
        dur = 40
        kept_edge = detect_waves(_sine(3.9, dur=dur, fs=fs, phase=0.3), fs, (1, 3.9), trough=trough)
        kept_out = detect_waves(_sine(3.9 * 1.03, dur=dur, fs=fs, phase=0.3), fs, (1, 3.9), trough=trough)
        assert kept_edge['min_pos'].size >= int((dur - 8) * 3.9) - 2, fs
        assert kept_out['min_pos'].size == 0, fs


# ----------------------------------------------------------------------------------
# amplitudes, slopes, positions
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize('lo, hi, f0', [(0.5, 4, 2), (4, 8, 6), (8, 12, 10), (11, 16, 13)])
def test_amplitude_recovered_in_every_band(lo, hi, f0):
    # a unit sine has peak-to-peak 2.0 on the unfiltered signal; the filtered (band)
    # amplitude is scaled by the filter gain |H(f0)|^2
    dur = 30
    d = detect_waves(_sine(f0, amp=1.0, dur=dur), FS, fband=(lo, hi))
    assert d['min_pos'].size >= int((dur - 2 * 4 / lo) * f0) - 2
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
    # trough='band' uses the interpolated crossing: within 1 % at fs=200. The default
    # 'refine' keeps v1.0.0's crossing (first negative sample), which shortens down_dur by
    # up to one sample: up to +4 % at fs=200 (the same for every wave of a pure sine).
    band = (0.5, 0.9) if f < 1 else (0.5, 4) if f == 1.0 else (1.0, 3.9)
    x = _sine(f, amp=amp, dur=60, fs=fs, phase=0.3)
    d = detect_waves(x, fs, fband=band, trough='band')
    assert np.mean(d['downslope']) == pytest.approx(4 * amp * f, rel=0.01)
    assert np.mean(d['downslope_band']) > 0
    r = detect_waves(x, fs, fband=band)
    ratio = np.mean(r['downslope']) / (4 * amp * f)
    assert 0.995 <= ratio <= 1 / (1 - 4 * f / fs) + 0.005    # at most one sample shorter


def test_upslope_matches_analytic():
    d = detect_waves(_sine(2.0, amp=1.0, dur=20, fs=500), 500, fband=(1.0, 3.9))
    assert np.mean(d['upslope']) == pytest.approx(4 * 1.0 * 2.0, rel=0.02)


@pytest.mark.parametrize('band', [(0.5, 4), (0.5, 0.9), (4, 8)])
def test_positions_are_ordered_and_inside_their_half_waves(band):
    x = _pink(120 * FS, FS, seed=3)
    for trough in ('band', 'paper'):
        d = detect_waves(x, FS, band, trough=trough)
        assert d['min_pos'].size > (5 if trough == 'band' else 0), trough
        for sfx in ('', '_band'):
            assert np.all(d['zero_pos_frac'] < d['min_pos' + sfx])
            assert np.all(d['min_pos' + sfx] < d['max_pos' + sfx])
            assert np.all(d['max_pos' + sfx] <= d['end_pos'])
            assert np.all(d['down_dur' + sfx] > 0)
            assert np.all(np.isfinite(d['downslope' + sfx]))
        assert np.all(np.diff(d['min_pos']) > 0)
        np.testing.assert_array_equal(d['min_pos'], d['min_pos_band'])
    d = detect_waves(x, FS, band, trough='band')
    assert np.all(d['min_val_band'] < 0) and np.all(d['max_val_band'] > 0)


def test_refine_is_the_v1_placement():
    # trough='refine' (default): broadband extreme within +/- half a period of fband[1]
    # around the band extreme, NOT clamped to the half-wave; the downslope runs from the
    # first negative sample (zero_pos), as in v1.0.0. A broadband trough at or before it
    # gives a NaN downslope (left out of the slope features).
    x = _pink(120 * FS, FS, seed=3) + 3 * np.random.default_rng(1).standard_normal(120 * FS)
    band = (0.5, 4)
    d = detect_waves(x, FS, band)
    half = int(round(0.5 * FS / band[1]))
    assert np.all(np.abs(d['min_pos'] - d['min_pos_band']) <= half)
    assert np.all(np.abs(d['max_pos'] - d['max_pos_band']) <= half)
    hp = _drift_removed(x - x.mean(), FS, band[0], 'butter')
    for p, pb in zip(d['min_pos'], d['min_pos_band']):
        assert p == pb - half + np.argmin(hp[pb - half:pb + half + 1])
    dd = (d['min_pos'] - d['zero_pos']) / FS
    np.testing.assert_allclose(d['down_dur'], dd)
    np.testing.assert_array_equal(np.isnan(d['downslope']), dd <= 0)
    # the band outputs always use the interpolated crossing
    np.testing.assert_allclose(d['down_dur_band'], (d['min_pos_band'] - d['zero_pos_frac']) / FS)
    # the gate uses the refined positions: every refined trough->peak lies in the band
    dt = d['delta_t']
    assert np.all((dt >= 0.99 / (2 * band[1]) - 1 / FS) & (dt <= 1.01 / (2 * band[0]) + 1 / FS))


def test_refine_trough_before_crossing_gives_nan_downslope():
    # a sharp broadband dip just BEFORE the band zero crossing, within half a period of
    # fband[1]: v1.0.0 placed the trough there (not clamped), so its downslope is NaN
    fs, f = 500, 3.5                                       # quarter period 71 ms < window 124 ms
    x = _sine(f, amp=50, dur=30, fs=fs, phase=0.0)
    t0, _ = _sine_wave_spans(f, 0.0, 30, fs)
    for t in t0[(t0 > 10) & (t0 < 17)]:
        i = int(round(t * fs)) - 10                        # 20 ms before the crossing
        x[i - 2:i + 3] -= 200
    d = detect_waves(x, fs, (0.5, 4))
    assert d['min_pos'].size > 5
    early = d['min_pos'] < d['zero_pos']
    assert early.sum() >= 10 and (~early).sum() >= 10
    assert np.all(np.isnan(d['downslope'][early]))
    got = _feats(WaveDetector(fs=fs, fband=(0.5, 4))(x, measure_on=x))
    assert np.isfinite(got['WAVE_SLOPE_MEAN'][0])        # NaN slopes are ignored


def test_band_mode_uses_interpolated_crossing():
    # sin(2 pi f t + phase): the down crossing sits between samples; the interpolated
    # position is exact to << 1 sample (kills a floor() regression)
    f, fs, phase = 2.0, 200, 0.37
    d = detect_waves(_sine(f, dur=30, fs=fs, phase=phase), fs, (0.5, 4), trough='band')
    t0, _ = _sine_wave_spans(f, phase, 30, fs)
    t0 = t0[np.searchsorted(t0, d['zero_pos_frac'] / fs - 0.1)]
    np.testing.assert_allclose(d['zero_pos_frac'] / fs, t0, atol=0.02 / fs)
    np.testing.assert_allclose(d['down_dur'], (d['min_pos'] - d['zero_pos_frac']) / fs)


def test_noise_bias_of_amplitudes_by_trough_mode():
    # 100 uV pk2pk 1 Hz sine + white noise SD 10 uV: 'band' (positions from the filtered
    # signal) is unbiased; 'refine' (v1.0.0) picks broadband noise extremes
    rng = np.random.default_rng(0)
    x = _sine(1.0, amp=50, dur=120) + 10 * rng.standard_normal(120 * FS)
    pb = np.mean(detect_waves(x, FS, (0.5, 4), trough='band')['pk2pk'])
    pr = np.mean(detect_waves(x, FS, (0.5, 4))['pk2pk'])
    assert pb == pytest.approx(100, rel=0.03)
    assert pr > 125


def test_measure_on_reads_amplitude_from_the_supplied_trace():
    t = np.arange(0, 60, 1 / 500)
    narrow = 50 * np.sin(2 * np.pi * 0.75 * t)
    broad = narrow + 8 * np.sin(2 * np.pi * 20 * t)
    base = detect_waves(narrow, 500, fband=(0.5, 0.9))
    on = detect_waves(narrow, 500, fband=(0.5, 0.9), measure_on=broad)
    # trough='refine' searches on the drift-removed x (v1.0.0), also with measure_on
    np.testing.assert_array_equal(base['min_pos'], on['min_pos'])
    np.testing.assert_allclose(on['min_val'], broad[on['min_pos']] - broad.mean())
    np.testing.assert_allclose(base['min_val_band'], on['min_val_band'])


# ----------------------------------------------------------------------------------
# gaps
# ----------------------------------------------------------------------------------
def test_single_nan_does_not_kill_the_record():
    # v1.0.0: one NaN -> WAVE_RATE 0 everywhere while DATA_RATE ~1. A 1-sample gap is
    # interpolated and needs no margin (gaps <= 25 ms, module docstring).
    x = _sine(2.0, amp=20, dur=120)
    x[50 * FS] = np.nan
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30, datarate=True)
    got = _feats(det(x))
    np.testing.assert_allclose(got['DATA_RATE'], [1, 1 - 1 / (30 * FS), 1, 1])
    np.testing.assert_allclose(got['WAVE_RATE'], 2.0, atol=0.03)
    d = det.detect(x)
    assert d['gaps'].tolist() == [[50 * FS, 50 * FS + 1]]
    # exactly the one wave whose span contains the NaN is gone
    clean = det.detect(_sine(2.0, amp=20, dur=120))
    assert clean['min_pos'].size - d['min_pos'].size == 1
    assert not np.any((d['zero_pos_frac'] <= 50 * FS) & (d['end_pos'] >= 50 * FS))


def test_default_gap_margin_depends_on_gap_length():
    m = _default_gap_margins(np.array([0.004, 0.025, 0.1, 0.5, 2.0, 4.0, 10.0]), 0.5, 250.0, 'refine')
    np.testing.assert_allclose(m, [0, 0, 2 * 4 * 0.05 ** 0.25, 2 * 4 * 0.25 ** 0.25, 2 * 4 * 1.0 ** 0.25, 8, 8])
    # paper mode: never beyond the support of its trace filters
    mp = _default_gap_margins(np.array([10.0]), 0.5, 500.0, 'paper')
    assert mp[0] == pytest.approx((999 + 12) / 500)


@pytest.mark.parametrize('every_s, L, min_kept', [(10, 0.02, 0.85), (30, 0.05, 0.7), (30, 1.0, 0.4)])
def test_dropouts_do_not_bias_wave_rate(every_s, L, min_kept):
    # R4/R5: 1 Hz sine (true rate 1.000) with regular dropouts. Short dropouts keep most
    # waves, and WAVE_RATE does not depend on the dropout density.
    fs, T = 250, 1800
    x = _sine(1.0, amp=50, dur=T, fs=fs, phase=0.4)
    rng = np.random.default_rng(3)
    for c in np.arange(every_s / 2, T, every_s):
        i = int((c + rng.uniform(-every_s / 4, every_s / 4)) * fs)
        x[i:i + max(1, int(L * fs))] = np.nan
    det = WaveDetector(fs=fs, fband=(0.5, 4), segm_size=30, datarate=True)
    got = _feats(det(x))
    kept = det.detect(x)['min_pos'].size / (T - 2)
    assert kept >= min_kept
    assert np.all(np.isfinite(got['WAVE_RATE']))
    # pooled rate (waves / analysable time) is unbiased
    pooled = np.nansum(got['WAVE_RATE'] * got['ANALYSABLE_RATE']) / np.nansum(got['ANALYSABLE_RATE'])
    assert pooled == pytest.approx(1.0, abs=0.01)
    assert np.all(got['ANALYSABLE_RATE'] <= got['DATA_RATE'] + 1e-12)


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
    ana = det._rate_mask(x.size, d).reshape(10, -1).sum(axis=1) / FS
    np.testing.assert_allclose(got['ANALYSABLE_RATE'], ana / 60)
    ok = ana > 10
    assert np.all(np.abs(got['WAVE_RATE'][ok] - 1.5) <= 1.5 / ana[ok])
    np.testing.assert_allclose(got['DATA_RATE'], np.isfinite(x).reshape(10, -1).mean(axis=1))
    # the wave count used per window equals the detections whose trough is in it
    n_in = np.histogram(d['min_pos'], bins=np.arange(11) * 60 * FS)[0]
    np.testing.assert_allclose(got['WAVE_RATE'][ok], (n_in / ana)[ok])


def test_gap_fill_is_the_spectral_fill(monkeypatch):
    # the gap fill is brainmaze_utils' fill_gaps with an explicit method (R1)
    calls = []
    real = wd.fill_gaps

    def spy(x, fs, **kw):
        calls.append(kw)
        return real(x, fs, **kw)

    monkeypatch.setattr(wd, 'fill_gaps', spy)
    x = _pink(120 * FS, FS, seed=2)
    x[40 * FS:42 * FS] = np.nan
    detect_waves(x, FS, (0.5, 4))
    assert calls == [{'method': 'spectral', 'max_interp_s': 0.1}]


def test_zero_filled_gap_would_change_waves_outside_the_margin():
    # a constant (e.g. zero) fill on an offset signal makes a step whose filter response
    # reaches beyond the margin; the spectral fill does not (waves outside the margin
    # match the gap-free run)
    x = _pink(240 * FS, FS, seed=2) + 200.0
    ref = detect_waves(x, FS, (0.5, 4))
    y = x.copy()
    y[100 * FS:104 * FS] = np.nan
    d = detect_waves(y, FS, (0.5, 4))
    j = np.isin(ref['min_pos'], d['min_pos'])
    k = np.isin(d['min_pos'], ref['min_pos'])
    assert k.mean() > 0.95
    np.testing.assert_allclose(d['min_val'][k], ref['min_val'][j], atol=1.0)


def test_gap_at_start_and_end_and_all_nan():
    x = _sine(2.0, dur=60)
    x[:3 * FS] = np.nan
    x[-2 * FS:] = np.nan
    d = detect_waves(x, FS, (0.5, 4))
    assert d['min_pos'].size > 0
    assert d['zero_pos_frac'].min() / FS >= 3 + 8
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
                  'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN', 'WAVE_SLOPE_MEDIAN']
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
            assert got['WAVE_SLOPE_MEDIAN'][i] == pytest.approx(np.nanmedian(d['upslope'][sel]))
            assert got['WAVE_SLOPE_MEDIAN_BAND'][i] == pytest.approx(np.median(d['upslope_band'][sel]))
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
    # 2 s windows, 8 s edge margin: the first four windows have no analysable time
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=2)
    rate = det(_sine(2.0, dur=30))[0][0]
    assert np.all(np.isnan(rate[:4])) and np.all(np.isnan(rate[-4:]))
    assert np.all(np.isfinite(rate[4:-4]))


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
    {'fs': FS, 'trough': 'v1'},
    {'fs': FS, 'trough': True},
    {'fs': FS, 'trough': 0},
    {'fs': FS, 'trough': None},
    {'fs': FS, 'trough': 'paper', 'features_on': 'both'},
    {'fs': 60, 'fband': (0.5, 4), 'trough': 'paper'},
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


def test_defaults_are_pinned():
    # changing any of these changes published numbers: they must be deliberate
    det = WaveDetector(fs=FS, fband=(0.5, 4))
    assert det.trough == 'refine'
    assert det.edge_margin_s == 8.0 and det.gap_margin_s is None
    assert (det.filter, det.filter_order, det.slope, det.nan_policy, det.features_on) == \
        ('butter', 2, 'downslope', 'fill', 'broadband')
    assert WaveDetector(fs=FS, fband=(4, 8), gap_margin_s=0.1).gap_margin_s == 0.1
    assert WaveDetector(fs=500, fband=(0.5, 4), trough='paper').edge_margin_s == pytest.approx(1011 / 500)
    import inspect
    p = inspect.signature(detect_waves).parameters
    assert p['trough'].default == 'refine' and p['filter'].default == 'butter'
    assert (wd._REFINE_HALF_WIN, wd._GATE_RTOL, wd._EDGE_MARGIN_PERIODS) == (0.5, 0.01, 4.0)
    assert (wd._GAP_MARGIN_SHORT_S, wd._GAP_MARGIN_SLOPE, wd._GAP_MARGIN_POWER,
            wd._GAP_MARGIN_MAX_PERIODS) == (0.025, 4.0, 0.25, 4.0)


@pytest.mark.parametrize('trough', [0, 0.5, True, None, float('nan'), 'Band'])
def test_detect_waves_validates_trough(trough):
    with pytest.raises(ValueError, match='trough'):
        detect_waves(_sine(2.0, dur=10), FS, (0.5, 4), trough=trough)


def test_paper_mode_rejects_measure_on():
    x = _sine(1.0, dur=20, fs=500)
    with pytest.raises(ValueError, match='measure_on'):
        detect_waves(x, 500, (0.5, 4), trough='paper', measure_on=x)


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


# ----------------------------------------------------------------------------------
# trough='paper' (Carvalho et al. 2024, Methods)
# ----------------------------------------------------------------------------------
def test_paper_trace_is_fir_then_moving_average():
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import firwin
    fs = 500
    x = _pink(60 * fs, fs, seed=4)
    taps = firwin(1999, [0.5, 35], pass_zero='bandpass', window='hamming', fs=fs)
    taps -= np.hamming(1999) * taps.sum() / np.hamming(1999).sum()   # DC gain -0.004 -> 0
    assert abs(taps.sum()) < 1e-12
    ref = uniform_filter1d(np.convolve(x, taps, mode='same'), 25, mode='nearest')
    np.testing.assert_allclose(_paper_trace(x, fs), ref, atol=1e-9)
    # zero phase: a 2 Hz sine comes out un-shifted (FIR gain ~1, 50 ms MA gain 0.97)
    t = np.arange(60 * fs) / fs
    y = _paper_trace(np.sin(2 * np.pi * 2 * t), fs)[10 * fs:-10 * fs]
    s = np.sin(2 * np.pi * 2 * t)[10 * fs:-10 * fs]
    g = np.sinc(2 * 0.05)                                    # MA gain at 2 Hz
    np.testing.assert_allclose(y, g * s, atol=0.01)


def _paper_loop(z, fs, band, thr=5.0):
    """Independent loop implementation of the paper's rule on the trace ``z``."""
    s = np.sign(z)
    s[s == 0] = 1
    down = np.flatnonzero((s[:-1] > 0) & (s[1:] < 0)) + 1
    up = np.flatnonzero((s[:-1] < 0) & (s[1:] > 0)) + 1
    out = []
    for dn in down:
        k = np.searchsorted(up, dn)
        if k >= up.size:
            continue
        u = up[k]
        zc = (dn - 1) + z[dn - 1] / (z[dn - 1] - z[dn])
        uc = (u - 1) - z[u - 1] / (z[u] - z[u - 1])
        if not (0.99 / (2 * band[1]) <= (uc - zc) / fs <= 1.01 / (2 * band[0])):
            continue
        tr = dn + int(np.argmin(z[dn:u]))
        if -z[tr] >= thr:
            out.append((tr, -z[tr] / ((tr - zc) / fs)))
    return np.array(out).reshape(-1, 2)


@pytest.mark.parametrize('band', [(0.5, 0.9), (1.0, 3.9)])
def test_paper_mode_matches_loop_reference(band):
    fs = 250
    x = _pink(600 * fs, fs, seed=11, rms=40) + _sine(0.7, amp=30, dur=600, fs=fs)
    det = WaveDetector(fs=fs, fband=band, trough='paper', amplitude_threshold=5,
                       edge_margin_s=0).detect(x)
    ref = _paper_loop(_paper_trace(x - x.mean(), fs), fs, band)
    # the detector drops only the very last half-wave (no following positive half-wave)
    assert ref.shape[0] - det['min_pos'].size in (0, 1)
    n = det['min_pos'].size
    assert n > 20
    np.testing.assert_array_equal(det['min_pos'], ref[:n, 0].astype(int))
    np.testing.assert_allclose(det['downslope'], ref[:n, 1], rtol=1e-9)


# ----------------------------------------------------------------------------------
# windows, median, threshold, context, memory
# ----------------------------------------------------------------------------------
def test_window_assignment_by_trough_end_exclusive():
    # a wave whose trough sits exactly on a window's first sample belongs to that window
    det = WaveDetector(fs=10, fband=(0.5, 4), segm_size=10, edge_margin_s=0, gap_margin_s=0)
    d = {k: np.array([], dtype=np.int64) for k in ('zero_pos', 'end_pos')}
    d.update({'min_pos': np.array([99, 100]), 'zero_pos_frac': np.array([97.0, 98.0]),
              'end_pos': np.array([102, 103]), 'gaps': np.empty((0, 2), dtype=np.int64)})
    for k in ('pk2pk', 'delta_t', 'min_val', 'max_val', 'downslope'):
        d[k] = np.array([1.0, 2.0])
    row = det._features_for_signal(np.zeros(300), None, d)
    np.testing.assert_array_equal(row['WAVE_PK2PK_MEAN'][:2], [1.0, 2.0])


def test_window_nanmedian_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    a[rng.integers(0, 500, 40)] = np.nan
    lo = np.sort(rng.integers(0, 500, 30))
    hi = np.minimum(lo + rng.integers(0, 60, 30), 500)
    got = _window_nanmedian(a, lo, hi)
    for g, l, h in zip(got, lo, hi):
        v = a[l:h][np.isfinite(a[l:h])]
        assert (np.isnan(g) and v.size == 0) or g == pytest.approx(np.median(v))


def test_amplitude_threshold_uses_the_broadband_trough():
    # band and broadband troughs differ: measure_on is the same wave at 1/20 amplitude, so
    # the band troughs (-50) pass a 5 uV threshold but the broadband ones (-2.5) do not
    x = _sine(0.75, amp=50, dur=60, fs=500)
    det = WaveDetector(fs=500, fband=(0.5, 0.9), amplitude_threshold=5)
    assert det.detect(x)['min_pos'].size > 10
    assert det.detect(x, measure_on=x / 20)['min_pos'].size == 0


@pytest.mark.parametrize('trough', ['refine', 'paper'])
def test_context_makes_consecutive_segments_match_the_whole_recording(trough):
    fs = 250
    x = _pink(900 * fs, fs, seed=21, rms=40)
    whole = WaveDetector(fs=fs, fband=(0.5, 4), trough=trough).detect(x)
    seg, ctx = 300 * fs, 20 * fs
    det = WaveDetector(fs=fs, fband=(0.5, 4), trough=trough)
    parts = []
    for k in range(3):
        a, b = max(0, k * seg - ctx), min(x.size, (k + 1) * seg + ctx)
        nb, na = k * seg - a, b - (k + 1) * seg
        d = det.detect(x[a:b], context=(nb, na))
        assert np.all((d['min_pos'] >= 0) & (d['min_pos'] < seg))
        parts.append(d['min_pos'] + k * seg)
    got = np.concatenate(parts)
    # inside the recording (away from its two real ends) every wave is found exactly once
    m = 30 * fs
    inner = lambda p: p[(p > m) & (p < x.size - m)]
    np.testing.assert_array_equal(inner(got), inner(whole['min_pos']))
    # __call__ with context: windows tile the core only
    v, names = WaveDetector(fs=fs, fband=(0.5, 4), segm_size=30)(x[:400 * fs], context=(50 * fs, 50 * fs))
    assert v[0].shape == (10,)
    with pytest.raises(ValueError, match='context'):
        det.detect(x[:100], context=(60, 50))
    with pytest.raises(ValueError, match='context'):
        det.detect(x[:100], context=(-1, 0))


def test_float32_2d_input_is_not_copied_up_front():
    # R7: rows are passed on as views; each is converted to float64 on its own
    X = np.zeros((4, 1000), dtype=np.float32)
    signals, _, _ = WaveDetector(fs=FS)._as_signal_list(X, None)
    assert all(np.shares_memory(s, X) and s.dtype == np.float32 for s in signals)
    X = _pink(4 * 60 * FS, FS, seed=1).reshape(4, -1).astype(np.float32)
    v32, _ = WaveDetector(fs=FS, segm_size=30)(X)
    v64, _ = WaveDetector(fs=FS, segm_size=30)(X.astype(np.float64))
    for a, b in zip(v32, v64):
        np.testing.assert_array_equal(a, b)


def test_datarate_adds_analysable_rate_last_and_keeps_v1_positions():
    det = WaveDetector(fs=FS, fband=(0.5, 4), segm_size=30, datarate=True)
    assert det.feature_names == ['DATA_RATE', 'WAVE_RATE', 'WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN',
                                 'WAVE_DELTA_T_MEAN', 'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN',
                                 'WAVE_SLOPE_MEDIAN', 'ANALYSABLE_RATE']
    got = _feats(det(_sine(2.0, dur=120)))
    # first / last windows lose the 8 s edge margin (+ the mean zc->trough / trough->end)
    assert got['ANALYSABLE_RATE'][1] == pytest.approx(1.0)
    assert got['ANALYSABLE_RATE'][0] == pytest.approx((22 - 0.125) / 30, abs=2 / (30 * FS))


# ----------------------------------------------------------------------------------
# golden values (R9): fixed expected outputs on deterministic input. A change here means
# the published numbers change -- update only deliberately, with the reason in the PR.
# Generated by brainmaze-work/scratch/eeg-wave/r2/gen_golden.py (numpy 1.26 and 2.x agree).
# ----------------------------------------------------------------------------------
def _golden_signal():
    """10 min at 250 Hz: 1/f background (30 uV RMS) + 0.7 Hz SO cycles + 2.5 Hz delta bursts."""
    fs = 250
    n = 600 * fs
    t = np.arange(n) / fs
    x = _pink(n, fs, rms=30.0, seed=1234)
    rng = np.random.default_rng(99)
    for c in rng.uniform(10, 590, 120):
        m = np.abs(t - c) < 0.5 / 0.7
        x[m] += -60 * np.cos(2 * np.pi * 0.7 * (t[m] - c))
    for c in rng.uniform(10, 590, 40):
        m = np.abs(t - c) < 1.5
        x[m] += 35 * np.sin(2 * np.pi * 2.5 * (t[m] - c)) * np.hanning(m.sum())
    return x, fs


GOLDEN = {
    ('default', (0.5, 0.9)): {
        'WAVE_RATE': [0.376061, 0.408333, 0.358333, 0.35, 0.288434],
        'WAVE_SLOPE_MEAN': [308.19435, 519.891963, 834.90742, 281.183311, 485.884897],
        'WAVE_SLOPE_MEDIAN': [200.926437, 243.050478, 323.617658, 248.124428, 276.718291],
        'WAVE_PK2PK_MEAN': [144.770865, 169.024946, 160.343625, 161.444741, 157.161801],
    },
    ('default', (1.0, 3.9)): {
        'WAVE_RATE': [1.596424, 1.508333, 1.566667, 1.433333, 1.651792],
        'WAVE_SLOPE_MEAN': [966.765457, 862.502361, 848.907204, 770.325563, 751.247591],
        'WAVE_SLOPE_MEDIAN': [496.051427, 519.014281, 540.227608, 485.080099, 626.183221],
        'WAVE_PK2PK_MEAN': [101.996514, 108.146227, 110.78014, 110.648088, 104.542695],
    },
    ('published', (0.5, 0.9)): {
        'WAVE_RATE': [0.376061, 0.408333, 0.358333, 0.35, 0.288434],
        'WAVE_SLOPE_MEAN': [263.558793, 338.899464, 563.084286, 199.566489, 302.349526],
        'WAVE_SLOPE_MEDIAN': [148.343909, 163.771052, 248.261967, 176.362424, 171.294407],
        'WAVE_PK2PK_MEAN': [100.530937, 118.361792, 117.94323, 118.002172, 107.28221],
    },
    ('published', (1.0, 3.9)): {
        'WAVE_RATE': [1.570536, 1.491667, 1.558333, 1.391667, 1.634496],
        'WAVE_SLOPE_MEAN': [641.130397, 573.974742, 563.97431, 520.444621, 529.675453],
        'WAVE_SLOPE_MEDIAN': [343.013431, 372.716025, 373.331022, 338.282025, 430.842703],
        'WAVE_PK2PK_MEAN': [71.31768, 75.91891, 78.431083, 78.570228, 73.122968],
    },
    ('band', (0.5, 0.9)): {
        'WAVE_RATE': [0.600186, 0.608333, 0.591667, 0.658333, 0.5588],
        'WAVE_SLOPE_MEAN': [56.985675, 73.083776, 76.120198, 70.644008, 59.881851],
        'WAVE_SLOPE_MEDIAN': [30.381507, 75.729207, 59.563786, 51.268781, 60.709177],
        'WAVE_PK2PK_MEAN': [49.087927, 60.98545, 58.080669, 57.617214, 45.202916],
    },
    ('band', (1.0, 3.9)): {
        'WAVE_RATE': [1.933101, 1.908333, 1.791667, 1.75, 2.031958],
        'WAVE_SLOPE_MEAN': [170.68875, 179.016632, 169.774231, 176.765934, 170.209284],
        'WAVE_SLOPE_MEDIAN': [149.497353, 163.442701, 164.283686, 176.379766, 197.149957],
        'WAVE_PK2PK_MEAN': [39.809503, 43.362645, 45.254423, 45.570808, 44.4297],
    },
    ('paper', (0.5, 0.9)): {
        'WAVE_RATE': [0.09348, 0.175, 0.158333, 0.166667, 0.144913],
        'WAVE_SLOPE_MEAN': [217.029852, 265.678338, 274.844802, 243.743935, 266.776949],
        'WAVE_SLOPE_MEDIAN': [227.786657, 211.466384, 261.296175, 207.270384, 203.222722],
        'WAVE_PK2PK_MEAN': [133.022454, 115.458256, 124.322902, 135.338886, 107.596409],
    },
    ('paper', (1.0, 3.9)): {
        'WAVE_RATE': [1.349012, 1.216667, 1.283333, 1.158333, 1.265543],
        'WAVE_SLOPE_MEAN': [314.690269, 325.595422, 372.155862, 325.532648, 317.149618],
        'WAVE_SLOPE_MEDIAN': [268.755309, 255.672384, 312.23228, 269.620851, 288.938055],
        'WAVE_PK2PK_MEAN': [45.814185, 50.173736, 53.060971, 49.305927, 50.741267],
    },
}


@pytest.mark.parametrize('key', sorted(GOLDEN, key=str))
def test_golden_values(key):
    from scipy.signal import filtfilt, firwin
    name, band = key
    x, fs = _golden_signal()
    kw = {'default': {}, 'published': dict(amplitude_threshold=5), 'band': dict(trough='band'),
          'paper': dict(trough='paper', amplitude_threshold=5)}[name]
    m = None
    if name == 'published':      # detect on x, measure on a 0.5-35 Hz trace (demo config)
        m = filtfilt(firwin(1001, [0.5, 35], pass_zero='bandpass', fs=fs, window='hamming'), 1.0, x)
    got = _feats(WaveDetector(fs=fs, fband=band, segm_size=120, **kw)(x, measure_on=m))
    for k, exp in GOLDEN[key].items():
        np.testing.assert_allclose(got[k], exp, rtol=1e-6, atol=1e-6, err_msg=f'{key} {k}')


@pytest.mark.parametrize('trough', ['refine', 'band'])
def test_truncated_last_half_wave_is_never_paired(trough):
    # the signal ends inside a positive half-wave: that half-wave has no closing zero
    # crossing, so the wave before it must not be reported even with edge_margin_s=0
    fs, f = 200, 2.0
    for phase in (0.3, 1.0, 2.0):
        x = _sine(f, dur=10.37, fs=fs, phase=phase)
        d = detect_waves(x, fs, (0.5, 4), trough=trough, edge_margin_s=0)
        assert d['end_pos'].max() < x.size - 1
        assert d['zero_pos'].min() > 0


def test_refined_trough_outside_the_span_is_checked_against_gaps():
    # trough='refine' can place the trough before the zero crossing (v1.0.0, not clamped).
    # A gap between that trough and the crossing must discard the wave, even with
    # gap_margin_s=0, because the trough value would come from the gap fill's neighbourhood.
    fs, f = 500, 3.5
    x = _sine(f, amp=50, dur=30, fs=fs)
    t0, _ = _sine_wave_spans(f, 0.0, 30, fs)
    for t in t0[(t0 > 8) & (t0 < 22)]:
        i = int(round(t * fs))
        x[i - 14] -= 300                                    # 1-sample dip 28 ms before the crossing
        x[i - 6:i - 4] = np.nan                             # 4 ms gap between dip and crossing
    d = detect_waves(x, fs, (0.5, 4), gap_margin_s=0)
    clean = detect_waves(np.where(np.isfinite(x), x, 0), fs, (0.5, 4), gap_margin_s=0)
    assert (clean['min_pos'] < clean['zero_pos']).sum() >= 40   # the dips do become troughs
    g = d['gaps']
    assert g.shape[0] >= 40
    lo = np.minimum(d['zero_pos_frac'], np.minimum(d['min_pos'], d['max_pos']))
    hi = np.maximum(d['end_pos'], np.maximum(d['min_pos'], d['max_pos']))
    overlap = (lo[:, None] < g[None, :, 1]) & (hi[:, None] >= g[None, :, 0])
    assert not overlap.any()
