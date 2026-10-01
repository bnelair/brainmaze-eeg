"""
Janca presets ('spike', 'ripple') of the single implementation, the ripple preset's filters,
resampling and validation, and the round-2 validation / resampling fixes (review of PR #67:
R3, R5, R6, R9, R11, R12).
"""

import warnings

import numpy as np
import pytest
from scipy.signal import resample_poly

from brainmaze_eeg.spikes import (JANCA_PRESETS, MAX_RESAMPLER_LOSS_DB, JancaDetector,
                                  SpikeDetectorHilbert, design_janca_filters,
                                  detect_spikes_janca, janca_params, janca_resampling,
                                  resampler_gain_db)
from brainmaze_eeg.spikes._filters import zero_phase_response_db as zp
from brainmaze_eeg.spikes.janca import _envelope, _largest_prime_factor, _rational
from brainmaze_eeg.tests.spike_synth import pink_background, synth_ieeg

EDGE_DB = 20 * np.log10(0.5)        # -6.02 dB: zero-phase Butterworth edge


def _tone(f, fs, dur):
    return np.sin(2 * np.pi * f * np.arange(int(dur * fs)) / fs)


def _pipeline_gain_db(f, fs, **kw):
    """Gain of a pure tone through the real detector path (filters + resampling + envelope)."""
    _, det = detect_spikes_janca(_tone(f, fs, 20.0), fs, return_details=True, **kw)
    env = det['envelope']
    return 20 * np.log10(np.median(env[len(env) // 4: 3 * len(env) // 4]))


# ------------------------------------------------------------------------------- presets
def test_spike_preset_is_the_eeg_forge_reference():
    assert dict(JANCA_PRESETS['spike']) == {
        'band': (10.0, 60.0), 'filter_order': 3, 'powerline': 50.0, 'notch_width': 5.0,
        'notch_order': 3, 'notch_harmonics': 1, 'target_fs': 200.0, 'decimation': 'integer',
        'window_s': 5.0, 'threshold': 3.65, 'min_distance_s': 0.1, 'eps_rel': 1e-6}
    assert janca_params() == janca_params('spike') == dict(JANCA_PRESETS['spike'])


def test_ripple_preset_differs_only_in_band_and_analysis_rate():
    s, r = janca_params('spike'), janca_params('ripple')
    assert {k for k in s if s[k] != r[k]} == {'band', 'target_fs'}
    assert r['band'] == (80.0, 250.0) and r['target_fs'] == 1000.0


def test_presets_are_read_only():
    with pytest.raises(TypeError):
        JANCA_PRESETS['spike']['threshold'] = 1.0
    with pytest.raises(TypeError):
        JANCA_PRESETS['new'] = {}


def test_overrides_on_top_of_a_preset():
    p = janca_params('ripple', threshold=4.0, notch_harmonics=5)
    assert p['threshold'] == 4.0 and p['notch_harmonics'] == 5 and p['band'] == (80.0, 250.0)
    with pytest.raises(ValueError, match='unknown Janca preset'):
        janca_params('hfo')
    with pytest.raises(TypeError, match='unknown Janca parameter'):
        janca_params('ripple', bandwidth=(80, 250))


def test_one_code_path_preset_equals_explicit_parameters():
    fs = 2000
    x, _ = synth_ieeg(fs, dur=30.0, seed=1, mains_hz=None)
    a = detect_spikes_janca(x, fs, preset='ripple')
    b = detect_spikes_janca(x, fs, band=(80, 250), target_fs=1000)     # spike preset + overrides
    np.testing.assert_array_equal(a, b)
    c = detect_spikes_janca(x, fs, preset='ripple', threshold=5.0)
    d = detect_spikes_janca(x, fs, band=(80, 250), target_fs=1000, threshold=5.0)
    np.testing.assert_array_equal(c, d)
    X = np.vstack([x, x[::-1]])
    for p, q in zip(JancaDetector('ripple', threshold=5.0).detect(X, fs),
                    detect_spikes_janca(X, fs, preset='ripple', threshold=5.0)):
        np.testing.assert_array_equal(p, q)
    assert repr(JancaDetector('ripple', threshold=5.0)) == "JancaDetector('ripple', threshold=5.0)"


def test_details_report_preset_and_resolved_params():
    x, _ = synth_ieeg(2000, dur=20.0, seed=0, mains_hz=None)
    _, info = detect_spikes_janca(x, 2000, preset='ripple', threshold=4.0, return_details=True)
    assert info['preset'] == 'ripple'
    assert info['params']['band'] == (80.0, 250.0) and info['params']['threshold'] == 4.0
    assert info['fs_analysis'] == 1000.0 and (info['up'], info['down']) == (1, 2)


# ---------------------------------------------------------------- ripple preset: filters
RIPPLE_FS = [1000, 1024, 1500, 2000, 2048, 5000, 10000, 32000]


@pytest.mark.parametrize('fs', RIPPLE_FS)
def test_ripple_preset_filter_design(fs):
    p = janca_params('ripple')
    f = design_janca_filters(fs, p['band'], p['filter_order'], p['powerline'],
                             p['notch_width'], p['notch_order'], p['notch_harmonics'])
    np.testing.assert_allclose(zp(f['bandpass'], [80.0, 250.0], fs), EDGE_DB, atol=0.01)
    assert zp(f['bandpass'], [140.0], fs)[0] > -0.1                 # pass band
    assert zp(f['bandpass'], [40.0], fs)[0] < -30                   # spike band suppressed
    assert zp(f['bandpass'], [500.0], fs)[0] < -30 if fs > 1000 else True


@pytest.mark.parametrize('fs', [1024, 2000, 2048, 5000, 32000])
@pytest.mark.parametrize('f', [80.0, 140.0, 250.0])
def test_ripple_preset_tone_gain_through_pipeline(fs, f):
    """End to end (filters + resampling + envelope): -6.02 dB at the edges, 0 dB inside."""
    want = EDGE_DB if f in (80.0, 250.0) else 0.0
    got = _pipeline_gain_db(f, fs, preset='ripple')
    assert got == pytest.approx(want, abs=MAX_RESAMPLER_LOSS_DB + 0.05)


@pytest.mark.parametrize('fs', [2000, 5000])
def test_ripple_preset_rejects_aliases(fs):
    """A tone above the analysis Nyquist does not alias into the ripple band."""
    up, down, fs_a = janca_resampling(fs, 1000.0)
    alias_src = fs_a - 200.0                       # would fold onto 200 Hz
    assert _pipeline_gain_db(alias_src, fs, preset='ripple') < -40


def test_ripple_preset_validation():
    x500, _ = synth_ieeg(500, dur=10.0, seed=0, mains_hz=None)
    with pytest.raises(ValueError, match='Nyquist'):
        detect_spikes_janca(x500, 500, preset='ripple')            # 250 Hz edge = Nyquist
    x2200, _ = synth_ieeg(2200, dur=10.0, seed=0, mains_hz=None)
    with pytest.raises(ValueError, match='anti-alias'):
        detect_spikes_janca(x2200, 2200, preset='ripple', target_fs=550)   # 250/275 Hz
    # at the input rate (no resampling) the filters are exact even close to Nyquist
    x600, _ = synth_ieeg(600, dur=10.0, seed=0, mains_hz=None)
    assert detect_spikes_janca(x600, 600, preset='ripple').dtype == np.int64


def test_ripple_preset_finds_synthetic_ripples():
    """Smoke test only: synthetic 120 Hz bursts in 1/f noise. NOT a validation on real ripples."""
    fs = 2000
    rng = np.random.default_rng(3)
    x = pink_background(int(60 * fs), fs, 30.0, rng)
    t = np.arange(int(0.06 * fs)) / fs
    burst = 150.0 * np.hanning(t.size) * np.sin(2 * np.pi * 120 * t)
    truth = np.arange(3.0, 57.0, 2.7)
    for s in truth:
        i = int(s * fs)
        x[i:i + t.size] += burst
    det = detect_spikes_janca(x, fs, preset='ripple', powerline=None)
    centres = (truth + 0.03) * fs
    hits = sum(bool(np.any(np.abs(det - c) <= 0.05 * fs)) for c in centres)
    assert hits >= 0.9 * truth.size
    assert det.size <= truth.size + 3


# ---------------------------------------------------- resampler edge check (review R3)
@pytest.mark.parametrize('fs,band,ok', [
    (1000, (10, 85), True), (1000, (10, 86), False), (1000, (80, 99), False),
    (1000, (60, 95), False), (2048, (10, 99), False), (2048, (10, 86), True),
    (2048, (80, 250), True)])
def test_band_edge_near_analysis_nyquist(fs, band, ok):
    x = np.random.default_rng(0).normal(size=int(20 * fs))
    kw = {'band': band, 'target_fs': 1000.0} if band[1] > 100 else {'band': band}
    if ok:
        detect_spikes_janca(x, fs, **kw)
    else:
        with pytest.raises(ValueError, match='anti-alias'):
            detect_spikes_janca(x, fs, **kw)


@pytest.mark.parametrize('fs,band', [(1000, (10, 85)), (2048, (10, 86)), (5000, (20, 85))])
def test_configured_band_just_under_the_limit_through_pipeline(fs, band):
    """A configured band at the limit is still -6 dB (+-0.1 dB) at its edges end to end."""
    got = _pipeline_gain_db(band[1], fs, band=band, powerline=None)
    assert got == pytest.approx(EDGE_DB, abs=MAX_RESAMPLER_LOSS_DB + 0.05)
    got = _pipeline_gain_db(band[0], fs, band=band, powerline=None)
    assert got == pytest.approx(EDGE_DB, abs=0.05)


@pytest.mark.parametrize('fs,up,down', [(1000, 1, 5), (2048, 1, 10), (24414.0625, 71, 8667)])
def test_resampler_gain_matches_resample_poly(fs, up, down):
    fs_a = fs * up / down
    for f in (0.5 * fs_a / 2, 0.86 * fs_a / 2, 0.95 * fs_a / 2):
        n = int(round(4 * fs / f)) * 50
        x = np.sin(2 * np.pi * f * np.arange(n) / fs)
        y = resample_poly(x, up, down)
        k = np.arange(y.size // 4, 3 * y.size // 4)            # away from the ends
        t = k / fs_a
        A = np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)])
        coef = np.linalg.lstsq(A, y[k], rcond=None)[0]           # amplitude of the tone
        got = 20 * np.log10(np.hypot(*coef))
        assert got == pytest.approx(resampler_gain_db(f, fs, up, down), abs=0.05)


# --------------------------------------------------------- rational resampling (R5)
@pytest.mark.parametrize('fs', [24414.0625, 511.99, 1000.0001, 2048.0, 500.0])
def test_rational_ratio_is_bounded_and_exact_when_simple(fs):
    up, down = _rational(200.0 / fs)
    assert abs(fs * up / down - 200.0) <= 1e-6 * 200.0
    if fs in (2048.0, 500.0):
        assert fs * up / down == 200.0                     # exact small ratios found exactly


@pytest.mark.parametrize('fs', [24414.0625, 511.99])
def test_real_world_rates_run_and_keep_timing(fs):
    x, truth = synth_ieeg(fs, dur=40.0, seed=3, amp_range=(250, 400), mains_hz=None)
    tol = 0.05 * fs
    for det in (detect_spikes_janca(x, fs, decimation='exact'), detect_spikes_janca(x, fs),
                SpikeDetectorHilbert().detect(x[None, :], fs)[0]):
        hits = sum(bool(np.any(np.abs(det - t) <= tol)) for t in truth)
        assert hits >= 0.85 * truth.size
    up, down, fs_a = janca_resampling(fs, 200.0, 'exact')
    assert fs_a == pytest.approx(200.0, rel=1e-6)


def test_v24_error_message_does_not_point_to_janca_option():
    with pytest.raises(ValueError) as e:
        _rational(np.pi / 1e7, who='SpikeDetectorHilbert(decimation=...)', max_den=1000)
    assert 'SpikeDetectorHilbert' in str(e.value) and "decimation='integer'" not in str(e.value)


# --------------------------------------------------------------- validation (R6, R12)
@pytest.mark.parametrize('kw', [
    dict(threshold=np.inf), dict(threshold=np.nan), dict(eps_rel=np.inf),
    dict(target_fs=np.inf), dict(powerline=np.inf), dict(window_s=np.inf),
    dict(min_distance_s=np.inf), dict(band=(10, np.inf)), dict(band=(np.nan, 60)),
    dict(filter_order=True), dict(filter_order=2.5), dict(filter_order=50),
    dict(notch_order=0), dict(notch_width=100), dict(decimation='exactly'),
    dict(threshold='3.65'), dict(window_s=-1)])
def test_janca_parameters_validated_at_construction(kw):
    with pytest.raises((ValueError, TypeError)):
        JancaDetector(**kw)
    with pytest.raises((ValueError, TypeError)):
        janca_params(**kw)


def test_janca_detector_validates_band_against_fs_at_detect():
    det = JancaDetector(band=(10, 300))             # valid without fs ...
    with pytest.raises(ValueError):
        det.detect(np.random.default_rng(0).normal(size=(1, 5000)), 500)   # ... not at 500 Hz


def test_janca_window_and_record_length_checks():
    x = np.random.default_rng(0).normal(size=5000)
    with pytest.raises(ValueError, match='window_s'):
        detect_spikes_janca(x, 500, window_s=1e-4)
    with pytest.raises(ValueError, match='too short'):
        detect_spikes_janca(x[:20], 500)


@pytest.mark.parametrize('kw', [
    dict(k1=-1), dict(k1=np.nan), dict(k3=np.nan), dict(winsize=0), dict(noverlap=5),
    dict(noverlap=6), dict(noverlap=-1), dict(buffering=0), dict(buffering=-1),
    dict(discharge_tol=-1), dict(polyspike_union_time=-1), dict(f_type=4),
    dict(cheb_rs=3.0), dict(cheb_transition_hz=(0, 10)), dict(bandwidth=(10, np.inf)),
    dict(decimation=-200), dict(main_hum_freq=np.nan), dict(_CHEB_HUM_R_AT_200=0.5),
    dict(run=1)])
def test_v24_parameters_validated_at_construction(kw):
    with pytest.raises((ValueError, TypeError)):
        SpikeDetectorHilbert(**kw)


def test_v24_attributes_changed_later_are_validated_at_run():
    det = SpikeDetectorHilbert()
    det.k1 = np.nan
    with pytest.raises(ValueError, match='k1'):
        det.run(np.random.default_rng(0).normal(size=(5000, 1)), 512)


def test_v24_band_must_be_below_input_nyquist_when_upsampling():
    x = np.random.default_rng(0).normal(size=(3000, 1))
    with pytest.raises(ValueError, match='input'):
        SpikeDetectorHilbert().run(x, 100.0)
    SpikeDetectorHilbert(bandwidth=(10, 40)).run(x, 100.0)      # well below 50 Hz: fine
    with pytest.raises(ValueError, match='anti-alias'):           # 0.9 x input Nyquist
        SpikeDetectorHilbert(bandwidth=(10, 45)).run(x, 100.0)


# ------------------------------------------------------------- v24 filter family (R11)
@pytest.mark.parametrize('fs,dec', [(2000, 1000), (1000, 0), (5000, 500)])
def test_v24_keeps_chebyshev_at_any_analysis_rate(fs, dec):
    det = SpikeDetectorHilbert(decimation=dec)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        filt = det.design_filters(dec or fs)
    assert filt['f_type'] == 1
    np.testing.assert_allclose(zp(filt['bandpass'], [10.0, 60.0], dec or fs), -12.0, atol=0.05)


# ---------------------------------------------------------------- Hilbert length (R9)
def test_hilbert_padding_only_for_slow_lengths():
    from scipy.signal import hilbert
    rng = np.random.default_rng(0)
    for n in (12000, 6087500 // 25, 3 * 5 * 7 * 11 * 13):          # smooth lengths: unpadded
        x = rng.normal(size=n)
        assert _largest_prime_factor(n) <= 1000
        np.testing.assert_array_equal(_envelope(x), np.abs(hilbert(x)))
    n = 100003                                                      # prime: padded
    assert _largest_prime_factor(n) == n
    x = rng.normal(size=n)
    a, b = _envelope(x), np.abs(hilbert(x))
    core = slice(n // 10, -n // 10)
    assert np.max(np.abs(a[core] - b[core])) < 1e-2 * np.median(b)
    assert [_largest_prime_factor(k) for k in (1, 2, 12, 97 * 89, 2 ** 20)] == [1, 2, 3, 97, 2]


# ---------------------------------------------------------------- memory layout (R8)
def test_float32_input_identical_and_channels_independent():
    fs = 1000
    X = np.vstack([synth_ieeg(fs, dur=30.0, seed=s)[0] for s in range(3)])
    a = detect_spikes_janca(X, fs)
    b = detect_spikes_janca(X.astype(np.float32), fs)
    c = [detect_spikes_janca(X[k].astype(np.float32).astype(np.float64), fs) for k in range(3)]
    for p, q, r in zip(a, b, c):
        np.testing.assert_array_equal(q, r)
        assert abs(p.size - q.size) <= 1                        # float32 rounding only
