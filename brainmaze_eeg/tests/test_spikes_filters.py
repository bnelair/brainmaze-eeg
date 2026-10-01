"""
Filter verification for the spike detectors.

Every filter used by every detector is designed from its (user-configurable) parameters at
sampling rates from 200 Hz to 32 kHz, and its *actual* response is checked:

- the zero-phase magnitude response (``|H|**2`` of the ``sos``, i.e. what ``sosfiltfilt``
  applies) at the stated edges, in the pass band and in the stop bands;
- stability (all poles strictly inside the unit circle) of the ``sos`` sections;
- zero phase: the impulse response of the forward-backward filter is symmetric;
- end to end: pure tones through each detector's real preprocessing path come out with the
  predicted gain (decimation included).

The edge levels: a Butterworth filter is -3.01 dB at its edges in a single pass, hence
-6.02 dB applied forward-backward. The v24 Chebyshev-II design is specified by a maximum
pass-band loss ``rp`` (6 dB, single pass) at the band edges, hence -12 dB zero-phase there.
"""

import numpy as np
import pytest
from scipy.signal import butter, sos2zpk, sosfiltfilt

from brainmaze_eeg.spikes._filters import (butter_bandpass, check_band, cheby2_lowpass,
                                           zero_phase_response_db as zp)
from brainmaze_eeg.spikes.barkmeier import design_barkmeier_filters
from brainmaze_eeg.spikes.janca import (SpikeDetectorHilbert, design_janca_filters,
                                        detect_spikes_janca)

FS_ALL = [200, 250, 256, 500, 512, 1000, 2000, 2048, 5000, 8000, 32000]
BUTTER_EDGE_DB = 20 * np.log10(0.5)          # -6.0206 dB: (-3.01 dB single pass) * 2
EDGE_TOL_DB = 0.01


def _max_pole(sos):
    if isinstance(sos, (list, tuple)):
        return max(_max_pole(s) for s in sos)
    return float(np.abs(sos2zpk(sos)[1]).max())


def _crossings(sos, fs, level, f_max):
    f = np.linspace(0.05, f_max, 200001)
    h = zp(sos, f, fs) - level
    i = np.flatnonzero(np.diff(np.sign(h)) != 0)
    return f[i] - h[i] * (f[i + 1] - f[i]) / (h[i + 1] - h[i])


# --------------------------------------------------------------------- detect_spikes_janca
@pytest.mark.parametrize('fs', FS_ALL)
def test_janca_bandpass_response(fs):
    sos = design_janca_filters(fs)['bandpass']
    assert _max_pole(sos) < 1
    assert zp(sos, [10, 60], fs) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=EDGE_TOL_DB)
    # only two -6 dB crossings, exactly at the configured edges
    np.testing.assert_allclose(_crossings(sos, fs, BUTTER_EDGE_DB, 0.99 * fs / 2), [10, 60],
                               atol=0.01)
    assert zp(sos, [np.sqrt(10 * 60)], fs)[0] == pytest.approx(0, abs=0.01)
    assert np.all(zp(sos, [15, 20, 30, 40], fs) > -0.6)
    assert zp(sos, [2], fs)[0] < -85


@pytest.mark.parametrize('fs', FS_ALL)
def test_janca_notch_response(fs):
    f = design_janca_filters(fs)
    assert [c for c, _ in f['notches']] == [50.0]
    sos = f['notches'][0][1]
    assert _max_pole(sos) < 1
    assert zp(sos, [47.5, 52.5], fs) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=EDGE_TOL_DB)
    assert zp(sos, [50], fs)[0] < -150
    assert np.all(zp(sos, [10, 30, 40, 60], fs) > -0.01)


@pytest.mark.parametrize('fs', [500, 1000, 5000, 32000])
@pytest.mark.parametrize('band,order', [((10, 60), 3), ((5, 40), 4), ((80, 250), 3)])
def test_janca_bandpass_is_configurable(fs, band, order):
    if band[1] >= fs / 2:
        with pytest.raises(ValueError, match='Nyquist'):
            design_janca_filters(fs, band=band, filter_order=order, powerline=None)
        return
    sos = design_janca_filters(fs, band=band, filter_order=order, powerline=None)['bandpass']
    assert zp(sos, band, fs) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=EDGE_TOL_DB)
    assert sos.shape[0] == order          # band-pass of prototype order N has N sections


@pytest.mark.parametrize('powerline,width,harm,fs,expect', [
    (60.0, 4.0, 1, 1000, [60.0]),
    (50.0, 5.0, 3, 1000, [50.0, 100.0, 150.0]),
    (50.0, 5.0, 3, 250, [50.0, 100.0]),        # 150 Hz does not fit below Nyquist: skipped
])
def test_janca_notch_is_configurable(powerline, width, harm, fs, expect):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = design_janca_filters(fs, powerline=powerline, notch_width=width,
                                 notch_harmonics=harm)
    assert [c for c, _ in f['notches']] == expect
    assert bool(f['skipped_notches']) == (len(expect) < harm)
    assert any('skipped' in str(x.message) for x in w) == (len(expect) < harm)
    for c, sos in f['notches']:
        assert zp(sos, [c - width / 2, c + width / 2], fs) == pytest.approx(
            [BUTTER_EDGE_DB] * 2, abs=EDGE_TOL_DB)


def test_janca_powerline_none_has_no_notch():
    assert design_janca_filters(500, powerline=None)['notches'] == []


def test_eeg_forge_ba_filters_are_unstable_at_32k_but_ours_are_not():
    """Documents the reference defect fixed by the sos design (eeg_forge uses b, a)."""
    hfs = 32000 / 2
    _, a = butter(3, [47.5 / hfs, 52.5 / hfs], 'bandstop')
    assert np.abs(np.roots(a)).max() > 1          # b/a: pole outside the unit circle
    assert _max_pole(design_janca_filters(32000)['notches'][0][1]) < 1


# ------------------------------------------------------------------------------ Barkmeier
@pytest.mark.parametrize('fs', FS_ALL)
def test_barkmeier_filters_response(fs):
    f = design_barkmeier_filters(fs)
    for name, band in (('narrow', (20, 50)), ('broad', (1, 35))):
        sos = f[name]
        assert sos.shape[0] == 2                  # 2nd-order Butterworth prototype
        assert _max_pole(sos) < 1
        assert zp(sos, band, fs) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=EDGE_TOL_DB)
        np.testing.assert_allclose(_crossings(sos, fs, BUTTER_EDGE_DB, 0.99 * fs / 2), band,
                                   atol=0.01)
        assert zp(sos, [np.sqrt(band[0] * band[1])], fs)[0] == pytest.approx(0, abs=0.01)
    assert zp(f['broad'], [80], fs)[0] < -29       # the old 1-80 Hz band is now rejected
    assert zp(f['narrow'], [5], fs)[0] < -60


def test_barkmeier_bands_and_orders_are_configurable():
    f = design_barkmeier_filters(1000, narrow_band=(15, 45), broad_band=(0.5, 70),
                                 narrow_order=4, broad_order=3)
    assert zp(f['narrow'], [15, 45], 1000) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=0.01)
    assert zp(f['broad'], [0.5, 70], 1000) == pytest.approx([BUTTER_EDGE_DB] * 2, abs=0.01)
    assert f['narrow'].shape[0] == 4 and f['broad'].shape[0] == 3


# ------------------------------------------------------------------- SpikeDetectorHilbert
@pytest.mark.parametrize('fs', FS_ALL)
def test_v24_chebyshev_bandpass_response(fs):
    det = SpikeDetectorHilbert(decimation=0)
    f = det.design_filters(fs)
    assert f['f_type'] == 1
    bp = f['bandpass']
    assert _max_pole(bp) < 1
    # v24 spec: <= rp (6 dB) single-pass loss at the band edges -> -12 dB zero-phase,
    # >= rs (60 dB) single-pass attenuation 5 Hz below / 10 Hz above -> <= -120 dB.
    assert zp(bp, [10, 60], fs) == pytest.approx([-12.0, -12.0], abs=0.05)
    assert np.all(zp(bp, [5, 70], fs) <= -120 + 0.05)
    assert np.all(zp(bp, [15, 20, 30, 40, 50], fs) > -0.5)
    # the fixed defect: the old design passed the pass-band edge as cheby2's Wn (stop-band
    # edge) and attenuated 15 Hz / 50 Hz by -15 / -8 dB; now they are in the pass band
    assert zp(bp, [15], fs)[0] > -0.2 and zp(bp, [50], fs)[0] > -0.2


@pytest.mark.parametrize('fs', FS_ALL)
def test_v24_hum_notch_and_highpass(fs):
    f = SpikeDetectorHilbert(decimation=0).design_filters(fs)
    hp = f['highpass']
    assert zp(hp, [1.0], fs)[0] == pytest.approx(BUTTER_EDGE_DB, abs=EDGE_TOL_DB)
    assert zp(hp, [10.0], fs)[0] > -0.01
    c, sos = f['notches'][0]
    assert c == 50.0 and _max_pole(sos) < 1
    assert zp(sos, [50.0], fs)[0] < -100
    assert zp(sos, [0.0], fs)[0] == pytest.approx(0, abs=1e-6)
    # -3 dB (single pass) width ~0.955 Hz at every rate (v24: 0.985 pole radius at 200 Hz)
    ff = np.linspace(45, 55, 100001)
    below = ff[zp(sos, ff, fs) / 2 < -3.0103]
    assert below.max() - below.min() == pytest.approx(0.955, abs=0.02)
    assert np.all(zp(sos, [45, 55], fs) > -0.2)


def test_v24_notch_harmonics_up_to_band_top():
    f = SpikeDetectorHilbert(decimation=0, bandwidth=[80, 250], f_type=2).design_filters(2000)
    assert [c for c, _ in f['notches']] == [50.0, 100.0, 150.0, 200.0, 250.0]


@pytest.mark.parametrize('fs', [200, 1000])
def test_v24_butterworth_and_fir_families(fs):
    for ftype, edge in ((2, BUTTER_EDGE_DB), (3, 20 * np.log10(0.25))):
        f = SpikeDetectorHilbert(decimation=0, f_type=ftype).design_filters(fs)
        assert f['f_type'] == ftype
        if ftype == 2:
            got = zp(f['bandpass'], [10, 60], fs)
        else:
            from scipy.signal import freqz
            h = np.ones(2, dtype=complex)
            for _, taps in f['bandpass']:
                h *= freqz(taps, 1, worN=[10, 60], fs=fs)[1]
            got = 20 * np.log10(np.abs(h) ** 2)
        assert got == pytest.approx([edge, edge], abs=0.05)


def test_cheby2_uses_order_design_stopband_edge():
    # regression for the cheby2 Wn bug: pass-band loss <= rp at the pass-band edge
    sos = cheby2_lowpass(60, 70, 200, rp=6, rs=60)
    assert zp(sos, [60], 200)[0] / 2 == pytest.approx(-6, abs=0.05)
    assert zp(sos, [70], 200)[0] / 2 <= -60 + 0.05


# ---------------------------------------------------------------------- zero-phase property
@pytest.mark.parametrize('fs', [200, 2048, 32000])
def test_forward_backward_is_zero_phase(fs):
    n = int(20 * fs) | 1
    x = np.zeros(n)
    c = n // 2
    x[c] = 1.0
    sos_list = [design_janca_filters(fs)['bandpass'], design_janca_filters(fs)['notches'][0][1],
                design_barkmeier_filters(fs)['narrow'], design_barkmeier_filters(fs)['broad']]
    for sos in sos_list:
        y = sosfiltfilt(sos, x)
        k = np.arange(1, int(2 * fs))
        assert np.abs(y[c + k] - y[c - k]).max() < 1e-9 * np.abs(y).max()
        assert np.argmax(np.abs(y)) == c


# -------------------------------------------------------------------- end-to-end tone tests
def _tone(f, fs, dur):
    return np.sin(2 * np.pi * f * np.arange(int(dur * fs)) / fs)


@pytest.mark.parametrize('fs', [200, 256, 500, 2048, 5000])
@pytest.mark.parametrize('f', [5, 10, 12, 30, 55, 60, 80])
def test_janca_tone_gain_through_detector_pipeline(fs, f):
    if f >= 0.45 * fs:
        pytest.skip('tone above the band of this rate')
    x = _tone(f, fs, 20.0)
    _, det = detect_spikes_janca(x, fs, return_details=True)
    env = det['envelope']
    m = env[len(env) // 4: 3 * len(env) // 4]
    got = 20 * np.log10(np.median(m))
    filters = design_janca_filters(fs)
    want = zp([filters['bandpass']] + [s for _, s in filters['notches']], [f], fs)[0]
    if want > -40:
        assert got == pytest.approx(want, abs=0.15)
    else:
        assert got < -35
    if 12 <= f <= 45:                             # (55 Hz sits on the 50 Hz notch skirt, -3.3 dB)
        assert got > -2.0                         # inside the 10-60 Hz band: passed
    if f in (5, 80):
        assert got < -20                          # outside: suppressed


@pytest.mark.parametrize('fs', [200, 512, 2048])
@pytest.mark.parametrize('f', [2, 8, 30, 45, 70])
def test_barkmeier_tone_gain(fs, f):
    if f >= 0.45 * fs:
        pytest.skip('tone above the band of this rate')
    flt = design_barkmeier_filters(fs)
    x = _tone(f, fs, 20.0)
    for name in ('narrow', 'broad'):
        y = sosfiltfilt(flt[name], x)
        m = y[len(y) // 4: 3 * len(y) // 4]
        got = 20 * np.log10(np.sqrt(2) * m.std())
        want = zp(flt[name], [f], fs)[0]
        assert got == pytest.approx(want, abs=0.1) if want > -60 else got < -55


@pytest.mark.parametrize('fs', [200, 512, 2048])
@pytest.mark.parametrize('f', [5, 12, 30, 55, 70])
def test_v24_tone_gain_through_run(fs, f):
    x = _tone(f, fs, 30.0)[:, None]
    _, _, _, env, _, _ = SpikeDetectorHilbert().run(x, fs)
    m = env[len(env) // 4: 3 * len(env) // 4, 0]
    got = 20 * np.log10(np.median(m))
    filt = SpikeDetectorHilbert().design_filters(200.0)
    want = zp(list(filt['bandpass']) + [filt['highpass']] + [s for _, s in filt['notches']],
              [f], 200.0)[0]
    if want > -40:
        assert got == pytest.approx(want, abs=0.15)
    else:
        assert got < -35


# ------------------------------------------------------------------------------ validation
@pytest.mark.parametrize('band', [(0, 50), (-1, 50), (50, 20), (20, 20), (10, 100), (10, 120),
                                  (np.nan, 50), (10,), 'ab'])
def test_check_band_rejects_invalid(band):
    with pytest.raises(ValueError):
        check_band(band, 200)


@pytest.mark.parametrize('order', [0, -1, 2.5])
def test_invalid_order_raises(order):
    with pytest.raises(ValueError):
        butter_bandpass((10, 60), 500, order)
