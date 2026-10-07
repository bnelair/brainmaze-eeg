# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Janca (Hilbert-envelope) interictal spike detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interictal epileptiform discharge (IED) detection by modelling the distribution of the
band-passed signal's Hilbert envelope as log-normal and flagging envelope maxima that exceed
an adaptive threshold derived from that distribution:

    Janca, R., Jezdik, P., Cmejla, R., Tomasek, M., Worrell, G.A., Stead, M., Wagenaar, J.,
    Jefferys, J.G.R., Krsek, P., Komarek, V., Jiruska, P., Marusic, P. (2015). *Detection of
    Interictal Epileptiform Discharges Using Signal Envelope Distribution Modelling:
    Application to Epileptic and Non-Epileptic Intracranial Recordings.* Brain Topography
    28(1), 172-183. https://doi.org/10.1007/s10548-014-0379-1

Two implementations are provided:

:func:`detect_spikes_janca` (primary)
    The fast, streamlined formulation of the algorithm from **eeg_forge** by xnejed07
    (``eeg_forge/detection/spike_detection_janca.py``,
    https://gitlab.com/xnejed07/eeg_forge), reproduced faithfully (detection indices are
    identical on the parity data, see the README) with its numerical defects fixed and every
    parameter configurable. Multichannel layout ``(n_channels, n_samples)``. One
    implementation serves several frequency bands through named **presets** (see below).

:class:`SpikeDetectorHilbert` (MATLAB-v24 compatible)
    A port of the published MATLAB ``spike_detector_hilbert_v24.m`` with its full output
    (per-detection CDF/PDF weights, multichannel discharge grouping, ambiguous ``k2``
    class, segment buffering). Multichannel layout ``(n_samples, n_channels)`` (MATLAB's
    ``[samples, channels]``).

Presets of :func:`detect_spikes_janca` / :class:`JancaDetector`
---------------------------------------------------------------
A preset is a complete, named set of parameter values (:data:`JANCA_PRESETS`); any
parameter passed explicitly overrides the preset's value, e.g.
``detect_spikes_janca(x, fs, preset='ripple', threshold=4)``. There is one code path; a
preset only supplies defaults. :func:`janca_params` returns the resolved, validated values.

``'spike'`` (default)
    The eeg_forge reference values: band 10-60 Hz, analysis rate ``target_fs`` 200 Hz,
    50 Hz notch, 5 s window, threshold 3.65, 0.1 s minimum distance. Source: eeg_forge
    ``spike_detection_Janca`` (commit 54c3704) and Janca et al. (2015).
``'ripple'``
    Ripple band **80-250 Hz** (the clinical ripple band, e.g. Zijlmans, M., Jiruska, P.,
    Zelmann, R., Leijten, F.S.S., Jefferys, J.G.R., Gotman, J. (2012). *High-frequency
    oscillations as a new biomarker in epilepsy.* Annals of Neurology 71(2), 169-178.
    https://doi.org/10.1002/ana.22548) analysed at ``target_fs`` = **1000 Hz** [ours: four
    times the band top, so that after decimation the band top sits at half the analysis
    Nyquist; with the default ``decimation='integer'`` the analysis rate is
    ``fs / floor(fs / 1000)``, e.g. 1000 Hz for 2-5 kHz input, 1024 Hz for 2048 Hz input].
    Input of 501-1999 Hz is **not** decimated (``integer`` needs ``fs >= 2 * target_fs``)
    and is analysed at its own rate, where 250 Hz can lie close to Nyquist (0.98 of it at
    512 Hz, 0.998 at 501 Hz): the realised edges are still -6.02 dB, but there is almost no
    room above the band. Input of 500 Hz or less raises. Every other value (threshold,
    window, minimum distance, notch) is the ``'spike'`` preset's, carried over **untuned**.
    Applying the Janca envelope model to the ripple band is our choice; **this preset has
    not been validated on real ripples** (only its filters, resampling and validation are
    verified by the test-suite). Note that mains harmonics inside 80-250 Hz are not notched
    by default (``notch_harmonics=1`` notches only ``powerline``); pass e.g.
    ``notch_harmonics=5`` to notch 100-250 Hz (50 Hz mains) as well.

Algorithm of :func:`detect_spikes_janca`
----------------------------------------
For each channel, independently (all filters zero-phase, ``sosfiltfilt``, at the **input**
rate ``fs``):

1. Band-pass Butterworth, order ``filter_order`` (3), edges ``band`` (10, 60) Hz.
   ``band`` is the design edge, i.e. the -3 dB point of the single-pass filter; the
   forward-backward (zero-phase) application squares the response, so the realised
   response is **-6.02 dB at both edges** (as in eeg_forge and v24, which also run their
   designs forward-backward).
2. Band-stop Butterworth, order ``notch_order`` (3), ``powerline +/- notch_width/2``
   (50 +/- 2.5 Hz; -3 dB single pass, -6 dB zero-phase at those edges), optionally also at
   harmonics (``notch_harmonics``).
3. Resample with :func:`scipy.signal.resample_poly` (its anti-alias FIR runs on the
   already band-limited signal). ``decimation='integer'`` (default, reference): if
   ``target_fs`` is set and ``fs >= 2 * target_fs``, decimate by the **integer** factor
   ``q = floor(fs / target_fs)``; the analysis rate ``fs_a = fs / q`` is then generally
   *not* ``target_fs`` (500 Hz -> 250 Hz; 512 -> 256; 2048 -> 204.8; 256 or 399 Hz -> not
   decimated). ``decimation='exact'`` (ours): resample to ``target_fs`` whenever
   ``fs > target_fs`` by the rational factor ``up / down`` (see :func:`janca_resampling`;
   within a relative 1e-6 of ``target_fs``).
   The resampler's anti-alias filter attenuates the top ~15 % below the analysis Nyquist,
   so a resampled configuration is only accepted if that filter loses at most
   :data:`MAX_RESAMPLER_LOSS_DB` (0.1 dB) at ``band[1]`` (about ``band[1] <= 0.85 *
   fs_a / 2``); the realised high edge is then -6.02 to -6.12 dB (-6.12 dB at the limit,
   ``band[1]`` = 0.858 * ``fs_a / 2``).
4. Envelope ``e = |hilbert(x)|``. When the analysis length has a prime factor > 1000 the
   FFT is padded to :func:`scipy.fft.next_fast_len` (pocketfft is 3-11x slower on such
   lengths; measured). Padding changes the envelope near **both ends** of the record (the
   zeros wrap the end into the start): on band-passed noise, > 1 % relative change in the
   first ~0.2-0.27 s and the last ~0.05-0.2 s. Detections there can differ (on a 6.8 h
   record cut to a prime analysis length: 5 of ~1500, all in the last 1.6 s; none changed
   on a 1 h prime-length record), as they would for a record a few samples longer. All
   other lengths are transformed unpadded, exactly as in the reference.
5. Sliding statistics of ``L = log(e + eps)`` over a centred window of
   ``W = int(window_s * fs_a)`` samples (made odd), ``mode='reflect'`` at the ends::

       mu[n] = mean_{k in win(n)} L[k]
       sd[n] = sqrt( mean_{k in win(n)} (L[k] - mu[k])**2 )

   Note that ``sd`` subtracts the *per-sample* local mean ``mu[k]`` inside the window, not
   the window-centre mean ``mu[n]``; this is the reference definition and is kept as is
   (it is not the textbook moving standard deviation; for a stationary background the two
   agree to first order).
6. Log-normal mode and median: ``mode = exp(mu - sd**2)``, ``median = exp(mu)``;
   threshold ``T = threshold * (mode + median)``.
7. Detections are the maxima of ``e`` found by :func:`scipy.signal.find_peaks` with
   ``height=T`` and ``distance=int(min_distance_s * fs_a)`` samples; returned as sample
   indices of the **input** signal (``index_a * q``, resolution ``q`` input samples; with
   ``decimation='exact'``: ``round(index_a * down / up)``, i.e. mapped back with the
   realised ratio, so the rational approximation causes no timing drift).

Channels are processed one at a time, so the working memory is a few times one channel,
not a few times the whole montage.

Optional extensions (3.1.0; off by default, the steps above are then unchanged)
-------------------------------------------------------------------------------
- **Reference baseline** (``baseline=``, ``combine=``): step 6 with ``mu``, ``sd`` measured on
  a reference recording (:class:`~brainmaze_eeg.spikes.janca_baseline.JancaBaseline`)
  instead of the sliding window, or the lower / higher of the two thresholds. For channels
  that spike so often that the local window learns the spikes as background. The
  signal-path parameters (:data:`SIGNAL_PATH_PARAMS`) and the analysis rate must match the
  baseline's.
- **Gap-aware statistics** (``gap_aware_stats=True``): step 5 over valid samples only, with a
  minimum valid fraction per window, in O(n) for any window length (long windows on long
  records with dropouts); samples left out of the statistics are not detected either.

Known differences from the eeg_forge reference (all deliberate fixes)
---------------------------------------------------------------------
- **Filters in ``sos`` form.** The reference designs ``b, a`` transfer functions; at high
  sampling rates these lose precision (max complex response error of the 50 Hz band-stop
  5e-7 at 2 kHz, 5e-5 at 5 kHz, 2e-3 at 8 kHz, 7e-3 at 10 kHz, 6e-2 at 16 kHz) and the
  band-stop becomes **unstable** at 32 kHz (pole radius 1.0006) -> NaN -> silently no
  detections. The ``sos`` design has the same response where the ``b, a`` one is accurate:
  on the parity data the detection indices are identical at 200-10000 Hz; at 16 kHz one
  detection of 23 moves by one analysis sample; at 32 kHz the reference finds nothing.
- **Scale-invariant epsilon.** The reference adds an absolute ``1e-6`` to the envelope
  before the log. For data in volts (envelope ~1e-5..1e-8) that constant dominates the
  background, the threshold no longer adapts, and the detector returns nothing. Here
  ``eps = eps_rel * median(e)`` (``eps_rel = 1e-6``), so the result does not depend on the
  unit; at uV scale detections are unchanged.
- **Validation.** Every parameter is checked (type, finiteness, range) when it is resolved
  (:func:`janca_params`, also at :class:`JancaDetector` construction); band edges, notch
  frequencies, the analysis Nyquist, the resampler loss at the band edge, the window
  length and the record length are checked against ``fs`` at call time (``ValueError``).
  A power-line notch that does not fit below Nyquist is skipped with a ``UserWarning``
  (``powerline=None`` disables it silently).
- **Multichannel input** ``(n_channels, n_samples)`` (the reference is 1-D only).
- **Non-finite input raises** ``ValueError`` (the reference silently returns zero detections
  for a channel with a single NaN). For signals with gaps use
  :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` around
  :class:`JancaDetector`.
- ``min_distance_s * fs_a < 1`` is clamped to 1 sample (the reference would raise).
- **Optional rational resampling** (``decimation='exact'``); the default keeps the
  reference's integer decimation so that results are identical to it.
- **Hilbert FFT length** padded to a fast length only for lengths with a prime factor
  > 1000 (step 4); the parity fixtures and the 6.8 h parity recording are not affected.
"""

import functools
import math
import types
import warnings
from fractions import Fraction

import numpy as np
from scipy.fft import next_fast_len
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, firwin, hilbert, resample_poly, sosfiltfilt, filtfilt
from scipy.special import erf, ndtri

from brainmaze_eeg.spikes import _checks as chk
from brainmaze_eeg.spikes import _filters as flt
from brainmaze_eeg.spikes._gaps import FLAT_AS_GAP_S, flat_runs

__all__ = ['detect_spikes_janca', 'JancaDetector', 'JANCA_PRESETS', 'janca_params',
           'design_janca_filters', 'janca_decimation_factor', 'janca_resampling',
           'resampler_gain_db', 'MAX_RESAMPLER_LOSS_DB', 'janca_threshold',
           'SIGNAL_PATH_PARAMS', 'DECISION_PARAMS', 'COMBINE_MODES',
           'DEFAULT_MIN_VALID_FRACTION', 'DEFAULT_STATS_MARGIN_S', 'BASELINE_LEVEL_WARN_RATIO',
           'MONTAGE_LEVEL_WARN_RATIO', 'MONTAGE_LEVEL_QUANTILE', 'SpikeDetectorHilbert', 'spike_detector_hilbert_v24']


# ============================================================================ presets
_SPIKE = {
    'band': (10.0, 60.0), 'filter_order': 3, 'powerline': 50.0, 'notch_width': 5.0,
    'notch_order': 3, 'notch_harmonics': 1, 'target_fs': 200.0, 'decimation': 'integer',
    'window_s': 5.0, 'threshold': 3.65, 'min_distance_s': 0.1, 'eps_rel': 1e-6,
}
#: Named parameter sets of :func:`detect_spikes_janca` (read-only; see the module docstring
#: for their sources). ``'ripple'`` differs from ``'spike'`` only in ``band`` and
#: ``target_fs`` and is **not validated on real ripples**.
JANCA_PRESETS = types.MappingProxyType({
    'spike': types.MappingProxyType(dict(_SPIKE)),
    'ripple': types.MappingProxyType({**_SPIKE, 'band': (80.0, 250.0), 'target_fs': 1000.0}),
})

#: Maximum loss (dB) of the resampler's anti-alias filter allowed at ``band[1]``.
MAX_RESAMPLER_LOSS_DB = 0.1

# relative tolerance and denominator bound of the rational resampling ratio
_RATIO_REL_TOL = 1e-6
_RATIO_MAX_DEN = 2 ** 20


class _FromPreset:
    """Default of the :func:`detect_spikes_janca` parameters: take the preset's value."""

    def __repr__(self):
        return '<preset>'


_P = _FromPreset()


def janca_params(preset='spike', **overrides):
    """
    Resolved, validated parameters of :func:`detect_spikes_janca`.

    Parameters
    ----------
    preset : str
        Name of a preset in :data:`JANCA_PRESETS` (``'spike'`` or ``'ripple'``).
    **overrides
        Any parameter of :func:`detect_spikes_janca`; replaces the preset's value.

    Returns
    -------
    dict
        One value per parameter (``band`` ... ``eps_rel``), ready for
        ``detect_spikes_janca(x, fs, **params)``.

    Raises
    ------
    ValueError, TypeError
        Unknown preset or parameter, or a value of the wrong type / out of range (NaN and
        inf are rejected everywhere except ``target_fs=None`` / ``powerline=None``, which
        mean "off"). Checks that need the sampling rate happen in
        :func:`detect_spikes_janca`.
    """
    if not isinstance(preset, str) or preset not in JANCA_PRESETS:
        raise ValueError(f'unknown Janca preset {preset!r}; available: {sorted(JANCA_PRESETS)}')
    unknown = set(overrides) - set(_SPIKE)
    if unknown:
        raise TypeError(f'unknown Janca parameter(s) {sorted(unknown)}; '
                        f'available: {sorted(_SPIKE)}')
    p = dict(JANCA_PRESETS[preset])
    p.update({k: v for k, v in overrides.items() if v is not _P})
    p['band'] = chk.pair('band', p['band'], gt=0)
    p['filter_order'] = chk.integer('filter_order', p['filter_order'], ge=1, le=20)
    p['powerline'] = chk.number('powerline', p['powerline'], gt=0, allow_none=True)
    p['notch_width'] = chk.number('notch_width', p['notch_width'], gt=0)
    if p['powerline'] is not None and not p['notch_width'] < 2 * p['powerline']:
        raise ValueError(f"notch_width must be in (0, 2*powerline), got {p['notch_width']}")
    p['notch_order'] = chk.integer('notch_order', p['notch_order'], ge=1, le=20)
    p['notch_harmonics'] = chk.integer('notch_harmonics', p['notch_harmonics'], ge=1)
    p['target_fs'] = chk.number('target_fs', p['target_fs'], gt=0, allow_none=True)
    p['decimation'] = chk.choice('decimation', p['decimation'], ('integer', 'exact'))
    p['window_s'] = chk.number('window_s', p['window_s'], gt=0)
    p['threshold'] = chk.number('threshold', p['threshold'], gt=0)
    p['min_distance_s'] = chk.number('min_distance_s', p['min_distance_s'], ge=0)
    p['eps_rel'] = chk.number('eps_rel', p['eps_rel'], ge=0)
    return p


# ============================================================================ resampling
def janca_decimation_factor(fs, target_fs=200.0):
    """
    Integer decimation factor of the reference rule (``decimation='integer'``).

    ``floor(fs / target_fs)`` when ``target_fs`` is set and ``fs >= 2 * target_fs``,
    otherwise 1 (no decimation). With the default ``target_fs=200`` this is the eeg_forge rule
    (decimate only when ``fs >= 400``). The resulting analysis rate ``fs / q`` is generally
    **not** ``target_fs`` (512 Hz -> 256 Hz, 2048 Hz -> 204.8 Hz, 399 Hz -> 399 Hz).
    """
    fs = chk.number('fs', fs, gt=0)
    target_fs = chk.number('target_fs', target_fs, gt=0, allow_none=True)
    if target_fs is None:
        return 1
    if fs >= 2 * target_fs:
        return int(np.floor(fs / target_fs))
    return 1


def _rational(ratio, who='decimation', rel_tol=_RATIO_REL_TOL, max_den=_RATIO_MAX_DEN):
    """
    ``(up, down)`` with ``|up/down - ratio| <= rel_tol * ratio`` and the smallest such
    denominator among the continued-fraction convergents of ``ratio``.

    Exact ratios with a small denominator (500 -> 200 Hz: 2/5; 2048 -> 200 Hz: 25/256) are
    found exactly; awkward ones (24414.0625 -> 200 Hz, 511.99 -> 200 Hz) are approximated to
    ``rel_tol`` (1e-6). The caller maps detections back with the realised ratio, so the
    approximation shifts the analysis rate by at most ``rel_tol`` (0.0002 Hz at 200 Hz) but
    causes no timing drift.
    """
    x = Fraction(ratio)
    h0, h1, k0, k1 = 0, 1, 1, 0
    a = x
    while True:
        ai = math.floor(a)
        h0, h1 = h1, ai * h1 + h0
        k0, k1 = k1, ai * k1 + k0
        if k1 > max_den:
            break
        if h1 > 0 and abs(h1 / k1 - ratio) <= rel_tol * ratio:
            return int(h1), int(k1)
        frac = a - ai
        if frac == 0:
            break
        a = 1 / frac
    raise ValueError(f'{who}: cannot approximate the resampling ratio {ratio!r} to a relative '
                     f'{rel_tol} with a denominator <= {max_den}')


def janca_resampling(fs, target_fs=200.0, decimation='integer'):
    """
    Resampling applied by :func:`detect_spikes_janca` after filtering.

    Parameters
    ----------
    fs : float
        Input sampling rate (Hz).
    target_fs : float or None
        Decimation target (Hz); ``None`` keeps the input rate.
    decimation : {'integer', 'exact'}
        ``'integer'`` (reference): decimate by ``q = floor(fs / target_fs)`` when
        ``fs >= 2 * target_fs`` (see :func:`janca_decimation_factor`).
        ``'exact'``: rational resampling to ``target_fs`` whenever ``fs > target_fs``
        (never upsamples). ``up / down`` is the smallest-denominator continued-fraction
        convergent of ``target_fs / fs`` within a relative 1e-6, so any real rate works
        (TDT 24414.0625 Hz, rates estimated from timestamps such as 511.99 Hz).

    Returns
    -------
    (up, down, fs_analysis) : (int, int, float)
        :func:`scipy.signal.resample_poly` factors and the realised analysis rate
        ``fs * up / down`` (equal to ``target_fs`` to within 1e-6 relative with
        ``'exact'``).
    """
    fs = chk.number('fs', fs, gt=0)
    decimation = chk.choice('decimation', decimation, ('integer', 'exact'))
    target_fs = chk.number('target_fs', target_fs, gt=0, allow_none=True)
    if decimation == 'integer' or target_fs is None:
        q = janca_decimation_factor(fs, target_fs)
        return 1, q, fs / q
    if fs <= target_fs:
        return 1, 1, float(fs)
    up, down = _rational(target_fs / fs, who="decimation='exact'")
    return up, down, fs * up / down


def resampler_gain_db(freq, fs, up, down):
    """
    Gain (dB) at ``freq`` Hz of the anti-alias FIR that :func:`scipy.signal.resample_poly`
    designs by default for ``up / down`` at input rate ``fs`` (Kaiser window, beta 5, cutoff
    at the lower of the two Nyquist frequencies, ``20 * max(up, down) + 1`` taps; scipy's
    own design rule). 0 when no resampling happens.
    """
    g = math.gcd(int(up), int(down))
    up, down = int(up) // g, int(down) // g
    if up == down == 1:
        return 0.0
    max_rate = max(up, down)
    h = firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=('kaiser', 5.0))
    w = 2 * np.pi * float(freq) / (float(fs) * up)
    resp = np.abs(np.dot(h, np.exp(-1j * w * np.arange(h.size))))
    with np.errstate(divide='ignore'):
        return float(20 * np.log10(resp))


def _check_resampler_edge(high, fs, up, down, fs_a, name='band'):
    nyq = min(fs, fs_a) / 2            # the anti-alias cutoff is the lower Nyquist
    loss = -resampler_gain_db(high, fs, up, down)
    if loss > MAX_RESAMPLER_LOSS_DB:
        raise ValueError(
            f'{name} high edge ({high} Hz) is too close to the resampling Nyquist frequency '
            f'({nyq:g} Hz; fs={fs:g} Hz resampled by {up}/{down} to {fs_a:g} Hz): the '
            f'resampler\'s anti-alias filter attenuates it by {loss:.2f} dB, so the realised '
            f'edge would be {-6.02 - loss:.1f} dB instead of -6.0 dB (allowed loss '
            f'{MAX_RESAMPLER_LOSS_DB} dB, i.e. about {name}[1] <= {0.85 * nyq:.4g} Hz). '
            'Raise target_fs, or set target_fs=None to analyse at the input rate.')


# ============================================================================ filters
def design_janca_filters(fs, band=(10.0, 60.0), filter_order=3, powerline=50.0,
                         notch_width=5.0, notch_order=3, notch_harmonics=1):
    """
    Design the filters :func:`detect_spikes_janca` applies at the input rate ``fs``.

    Parameters are those of :func:`detect_spikes_janca`. ``band`` and the band-stop edges
    are design edges (-3 dB single pass); the detector applies the filters forward-backward,
    so the realised response is -6.02 dB there.

    Returns
    -------
    dict
        ``'bandpass'``: ``sos`` of the band-pass; ``'notches'``: list of
        ``(centre_hz, sos)`` band-stops actually applied; ``'skipped_notches'``: centres
        (Hz) that did not fit below Nyquist and were skipped (a ``UserWarning`` is issued).

    Raises
    ------
    ValueError
        Invalid band, order, notch width or harmonic count.
    """
    fs = chk.number('fs', fs, gt=0)
    filter_order = chk.integer('filter_order', filter_order, ge=1, le=20)
    out = {'bandpass': flt.butter_bandpass(band, fs, filter_order), 'notches': [],
           'skipped_notches': []}
    if powerline is None:
        return out
    powerline = chk.number('powerline', powerline, gt=0)
    notch_width = chk.number('notch_width', notch_width, gt=0)
    if not notch_width < 2 * powerline:
        raise ValueError(f'notch_width must be in (0, 2*powerline), got {notch_width}')
    notch_order = chk.integer('notch_order', notch_order, ge=1, le=20)
    notch_harmonics = chk.integer('notch_harmonics', notch_harmonics, ge=1)
    for k in range(1, notch_harmonics + 1):
        f0 = k * powerline
        stop = (f0 - notch_width / 2.0, f0 + notch_width / 2.0)
        if stop[1] >= fs / 2:
            out['skipped_notches'].append(f0)
            continue
        out['notches'].append((f0, flt.butter_bandstop(stop, fs, notch_order)))
    if out['skipped_notches']:
        warnings.warn(f'power-line notch(es) at {out["skipped_notches"]} Hz (width '
                      f'{notch_width} Hz) do not fit below Nyquist ({fs / 2} Hz) and were '
                      'skipped', UserWarning, stacklevel=2)
    return out


def _sos_padlen(sos):
    """Default ``padlen`` of :func:`scipy.signal.sosfiltfilt` for ``sos``."""
    ntaps = 2 * sos.shape[0] + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * ntaps


def _as_channels_first(x, name='x'):
    x = np.asarray(x)
    if np.iscomplexobj(x) or not (np.issubdtype(x.dtype, np.number)
                                  or np.issubdtype(x.dtype, np.bool_)):
        raise TypeError(f"'{name}' must be a real numeric array, got dtype {x.dtype}")
    if x.ndim == 1:
        return x[np.newaxis, :], True
    if x.ndim != 2:
        raise ValueError(f"'{name}' must be 1-D (n_samples,) or 2-D (n_channels, n_samples), "
                         f'got {x.ndim}-D')
    if x.shape[0] > x.shape[1]:
        raise ValueError(f"'{name}' has shape {x.shape}: more channels than samples. The "
                         'layout is (n_channels, n_samples); transpose your array (x.T).')
    return x, False


#: The Hilbert FFT is padded to :func:`scipy.fft.next_fast_len` only when the length's
#: largest prime factor exceeds this (there pocketfft is 3-11x slower; measured).
_FFT_PAD_PRIME = 1000


@functools.lru_cache(maxsize=64)
def _largest_prime_factor(n):
    n = int(n)
    if n < 2:
        return n
    lp, f = 1, 2
    while f * f <= n:
        while n % f == 0:
            lp, n = f, n // f
        f += 1 if f == 2 else 2
    return max(lp, n) if n > 1 else lp


def _envelope(y):
    """
    ``|hilbert(y)|``. For lengths with a large prime factor (> 1000) the FFT is padded to a
    fast length: 3-11x faster, and the envelope changes near both ends of the record
    (detections in the first/last ~2 s may differ, as they would for a record a few samples
    longer). Other lengths -- including every eeg_forge parity case -- are transformed
    unpadded, so their results are unchanged.
    """
    n = y.shape[-1]
    if _largest_prime_factor(n) > _FFT_PAD_PRIME:
        return np.abs(hilbert(y, N=next_fast_len(n), axis=-1)[..., :n])
    return np.abs(hilbert(y, axis=-1))


# ============================================================================ signal path
#: Parameters of :func:`detect_spikes_janca` that define the **signal path** (filters,
#: resampling, envelope offset). A :class:`~brainmaze_eeg.spikes.janca_baseline.JancaBaseline`
#: may only be used for detection with the same values (and the same analysis rate
#: ``fs_analysis``); :func:`detect_spikes_janca` raises ``ValueError`` naming every mismatch.
SIGNAL_PATH_PARAMS = ('band', 'filter_order', 'powerline', 'notch_width', 'notch_order',
                      'notch_harmonics', 'target_fs', 'decimation', 'eps_rel')
#: Decision parameters of :func:`detect_spikes_janca`: free to differ from a baseline's.
DECISION_PARAMS = ('threshold', 'min_distance_s', 'window_s')
#: How a reference baseline's threshold is combined with the local one (``combine``).
COMBINE_MODES = ('reference', 'min', 'max')
#: Default ``min_valid_fraction`` of ``gap_aware_stats=True``: a sliding window needs at
#: least this fraction of its samples valid, otherwise the local threshold is undefined
#: (NaN: no detections there).
DEFAULT_MIN_VALID_FRACTION = 0.5
#: Default ``stats_margin_s`` (s): analysis samples this close to an invalid sample (gap,
#: dropout, filled sample) are excluded from the background statistics, because filter and
#: Hilbert transients and the gap fill reach into the neighbouring signal (measured; see the
#: README, "Reference baseline").
DEFAULT_STATS_MARGIN_S = 0.5
#: A reference baseline whose envelope level (``exp(mu)``) differs from the record's by more
#: than this factor (either way) triggers a ``UserWarning``: units/gain/montage mismatch?
BASELINE_LEVEL_WARN_RATIO = 10.0
#: Second, montage-wide level check: the **median across channels** of the background level
#: ratio at the :data:`MONTAGE_LEVEL_QUANTILE` quantile of the log-envelope (record vs the
#: baseline model ``mu + ndtri(q) * sd``) beyond this factor (either way) triggers a
#: ``UserWarning``: gain or montage mismatch of about x3. A low quantile measures the
#: background between spikes, so dense spiking moves it much less than the mean-log
#: ``level_ratio`` (5 spikes/s of 600 uV in 30 uV background: x1.9 vs x4.4); the median
#: across channels ignores a minority of channels that spike or are off. Tuning and
#: evidence: README, "Reference baseline".
MONTAGE_LEVEL_WARN_RATIO = 3.0
#: Quantile of the log-envelope used by the montage-wide level check.
MONTAGE_LEVEL_QUANTILE = 0.1


def _signal_path(p, fs):
    """
    Filters and resampling of the resolved parameters ``p`` at input rate ``fs``, with the
    checks that need ``fs``. Returns ``(filters, up, down, fs_a, padlen)``.
    """
    filters = design_janca_filters(fs, p['band'], p['filter_order'], p['powerline'],
                                   p['notch_width'], p['notch_order'], p['notch_harmonics'])
    up, down, fs_a = janca_resampling(fs, p['target_fs'], p['decimation'])
    hi = p['band'][1]
    if hi >= fs_a / 2:
        raise ValueError(f'band high edge ({hi} Hz) must be < Nyquist of the analysis '
                         f'rate ({fs_a / 2} Hz; fs={fs} Hz resampled by {up}/{down}). Raise '
                         'target_fs or set target_fs=None.')
    if (up, down) != (1, 1):
        _check_resampler_edge(hi, fs, up, down, fs_a)
    padlen = max([_sos_padlen(filters['bandpass'])]
                 + [_sos_padlen(s) for _, s in filters['notches']])
    return filters, up, down, fs_a, padlen


def _channel_envelope(row, filters, up, down):
    """Band-pass, notch(es), resample and Hilbert envelope of one finite channel."""
    y = sosfiltfilt(filters['bandpass'], np.asarray(row, dtype=np.float64))
    for _, sos in filters['notches']:
        y = sosfiltfilt(sos, y)
    if (up, down) != (1, 1):
        y = resample_poly(y, up, down)
    return _envelope(y)


def _envelope_scale(e):
    """The reference's ``eps`` scale: ``median(e)``, or ``mean(e)`` if that is 0; 0 if both are."""
    scale = np.median(e) if e.size else 0.0
    if not scale > 0:
        scale = e.mean() if e.size else 0.0
    return scale if scale > 0 else 0.0


def janca_threshold(mu, sd, threshold):
    """
    Janca threshold ``threshold * (exp(mu - sd**2) + exp(mu))`` from the log-envelope mean
    ``mu`` and standard deviation ``sd``: ``threshold`` times the sum of the mode and the
    median of the log-normal envelope model, in the units of the input signal.
    """
    return threshold * (np.exp(mu - sd ** 2) + np.exp(mu))


def _sliding_sum(a, win):
    """Centred sliding sum over ``win`` (odd) samples, ends reflected like
    ``scipy.ndimage`` ``mode='reflect'`` (``d c b a | a b c d | d c b a``, repeated for
    windows longer than the record); O(n) time and memory via cumsum, for any ``win``."""
    h = win // 2
    n = a.size
    dtype = np.int64 if a.dtype.kind in 'bi' else np.float64
    if h <= n:
        ap = np.pad(a, h, mode='symmetric')
        cs = np.cumsum(ap, dtype=dtype)
        out = cs[win - 1:].copy()
        out[1:] -= cs[:-win]
        return out
    # Window longer than twice the record: the reflected extension is periodic with period
    # 2n (a, reversed a), so a window sum is whole periods plus a partial one; padding by h
    # would need O(win) memory.
    period = np.concatenate((a, a[::-1]))
    cs = np.concatenate((np.zeros(1, dtype=dtype), np.cumsum(period, dtype=dtype)))
    total = cs[-1]
    i = np.arange(n, dtype=np.int64)

    def prefix(m):                       # sum of the extension over [0, m) for any integer m
        q, r = np.divmod(m, 2 * n)
        return q * total + cs[r]

    return prefix(i + h + 1) - prefix(i - h)


def _masked_log_stats(log_e, ok, win, min_valid_fraction):
    """
    Sliding ``mu`` and ``sd`` of ``log_e`` over the valid samples (``ok``) of a centred
    window of ``win`` (odd) samples, ends reflected as in the reference
    (``uniform_filter1d(..., mode='reflect')``)::

        mu[n] = mean_{k in win(n), ok[k]} log_e[k]
        sd[n] = sqrt( mean_{k in win(n), ok[k]} (log_e[k] - mu[k])**2 )

    (the reference's definition, restricted to valid samples). Where fewer than
    ``min_valid_fraction * win`` samples of the window are valid, ``mu`` and ``sd`` are NaN.
    Without invalid samples this equals the reference to rounding (verified by the
    test-suite). O(n) in time and memory for any window length.
    """
    cnt = _sliding_sum(ok, win)
    centre = float(log_e[ok].mean())               # centred sums: no loss of precision
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = _sliding_sum(np.where(ok, log_e - centre, 0.0), win) / cnt + centre
        sd = np.sqrt(_sliding_sum(np.where(ok, (log_e - mu) ** 2, 0.0), win) / cnt)
    low = cnt < min_valid_fraction * win
    mu[low] = np.nan
    sd[low] = np.nan
    return mu, sd


def _local_threshold(e, threshold, eps_rel, win, ok=None, min_valid_fraction=None):
    """
    Local threshold curve of the reference algorithm (``ok is None``: exactly the eeg_forge
    computation), or, with a validity mask ``ok`` (``gap_aware_stats``), with the statistics
    taken over valid samples only (NaN where the window coverage is too low). NaN
    everywhere if the envelope is identically 0 (no detections, as in the reference).
    """
    if ok is None:
        scale = _envelope_scale(e)
        if not scale > 0:
            return np.full(e.shape, np.nan)
        log_e = np.log(e + eps_rel * scale)
        mu = uniform_filter1d(log_e, win, mode='reflect')
        sd = np.sqrt(uniform_filter1d((log_e - mu) ** 2, win, mode='reflect'))
        return threshold * (np.exp(mu - sd ** 2) + np.exp(mu))
    scale = _envelope_scale(e[ok])
    if not scale > 0:
        return np.full(e.shape, np.nan)
    log_e = np.log(e + eps_rel * scale)
    mu, sd = _masked_log_stats(log_e, ok, win, min_valid_fraction)
    return threshold * (np.exp(mu - sd ** 2) + np.exp(mu))


def _invalid_runs(valid_row):
    """``(starts, stops)`` of the runs of False in a boolean row."""
    d = np.diff(np.concatenate(([0], (~valid_row).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _analysis_ok(valid_row, up, down, n_a, margin_a):
    """
    Input-rate validity -> analysis-rate statistics mask: an analysis sample is excluded if it
    lies within ``margin_a`` analysis samples of the analysis-rate image
    ``[floor(a * up / down), ceil(b * up / down))`` of any invalid input run ``[a, b)``.
    """
    st, en = _invalid_runs(valid_row)
    if st.size == 0:
        return np.ones(n_a, dtype=bool)
    lo = np.clip(st * up // down - margin_a, 0, n_a)
    hi = np.clip(-(-en * up // down) + margin_a, 0, n_a)
    acc = np.zeros(n_a + 1, dtype=np.int64)
    np.add.at(acc, lo, 1)
    np.add.at(acc, hi, -1)
    return np.cumsum(acc[:-1]) == 0


def _stats_valid_mask(valid, X, fs, one_d):
    """``valid`` checked against ``X`` (bool, same shape), or derived from the flat-run rule."""
    if valid is None:
        V = np.ones(X.shape, dtype=bool)
        for c in range(X.shape[0]):
            for a, b in flat_runs(X[c], fs, FLAT_AS_GAP_S):
                V[c, a:b] = False
        return V
    V = np.asarray(valid)
    if V.dtype != np.bool_:
        raise TypeError(f'valid must be a boolean array (True = usable sample), got dtype '
                        f'{V.dtype}')
    if one_d and V.ndim == 1:
        V = V[np.newaxis, :]
    if V.shape != X.shape:
        raise ValueError(f'valid has shape {np.shape(valid)}; it must have the shape of x '
                         f'({X.shape if not one_d else X.shape[1:]})')
    return V


def _resolve_options(baseline, combine, broadcast_baseline, gap_aware_stats, valid,
                     min_valid_fraction, stats_margin_s):
    """Validate the baseline / gap-aware options; return ``(min_valid_fraction,
    stats_margin_s)`` with defaults filled in. Options that would be silently ignored raise."""
    from brainmaze_eeg.spikes.janca_baseline import JancaBaseline
    if baseline is not None and not isinstance(baseline, JancaBaseline):
        raise TypeError(f'baseline must be a JancaBaseline or None, got {type(baseline).__name__}')
    combine = chk.choice('combine', combine, COMBINE_MODES)
    if not isinstance(broadcast_baseline, (bool, np.bool_)):
        raise TypeError(f'broadcast_baseline must be a bool, got {broadcast_baseline!r}')
    if not isinstance(gap_aware_stats, (bool, np.bool_)):
        raise TypeError(f'gap_aware_stats must be a bool, got {gap_aware_stats!r}')
    if baseline is None and combine != 'reference':
        raise ValueError(f'combine={combine!r} needs a baseline (baseline=JancaBaseline(...))')
    if baseline is None and broadcast_baseline:
        raise ValueError('broadcast_baseline=True needs a baseline')
    if not gap_aware_stats:
        given = [n for n, v in (('valid', valid), ('min_valid_fraction', min_valid_fraction),
                                ('stats_margin_s', stats_margin_s)) if v is not None]
        if given:
            raise ValueError(f'{", ".join(given)} only apply with gap_aware_stats=True')
    if min_valid_fraction is None:
        min_valid_fraction = DEFAULT_MIN_VALID_FRACTION
    min_valid_fraction = chk.number('min_valid_fraction', min_valid_fraction, gt=0, le=1)
    if stats_margin_s is None:
        stats_margin_s = DEFAULT_STATS_MARGIN_S
    stats_margin_s = chk.number('stats_margin_s', stats_margin_s, ge=0)
    return combine, min_valid_fraction, stats_margin_s


def _baseline_rows(baseline, n_ch, broadcast_baseline):
    """Per-channel ``(mu, sd)`` of ``baseline`` for ``n_ch`` channels (channel-count rule)."""
    if baseline.n_channels == n_ch:
        return baseline.mu, baseline.sd
    if baseline.n_channels == 1 and broadcast_baseline:
        return np.repeat(baseline.mu, n_ch), np.repeat(baseline.sd, n_ch)
    hint = (' Pass broadcast_baseline=True to apply a 1-channel baseline to every channel.'
            if baseline.n_channels == 1 else
            ' Use baseline.select(...) to pick the rows matching x.')
    raise ValueError(f'the baseline has {baseline.n_channels} channel(s) but x has {n_ch}.'
                     + hint)


def _names_tuple(channel_names, n_ch=None):
    """``channel_names`` as a tuple of unique str (length ``n_ch`` if given), or ``None``."""
    if channel_names is None:
        return None
    if isinstance(channel_names, str):
        channel_names = [channel_names]
    names = tuple(channel_names)
    if not all(isinstance(c, str) for c in names) or len(set(names)) != len(names):
        raise ValueError(f'channel_names must be unique strings, got {channel_names!r}')
    if n_ch is not None and len(names) != n_ch:
        raise ValueError(f'channel_names has {len(names)} name(s) for {n_ch} channel(s) of x')
    return names


def _check_channel_names(baseline, names, broadcast_baseline):
    """
    Raise ``ValueError`` if the record's channel ``names`` (tuple) differ from the baseline's
    (order included). Not checkable (no-op) when the baseline has no names or a 1-channel
    baseline is broadcast.
    """
    bn = baseline.channel_names
    if bn is None or (baseline.n_channels == 1 and broadcast_baseline):
        return
    if names != bn:
        hint = (f' Use baseline.select({list(names)!r}) to reorder the baseline.'
                if set(names) <= set(bn) else '')
        raise ValueError(f'the channels of x {list(names)} do not match the baseline\'s '
                         f'{list(bn)} (names and order must be equal: row i of x uses row i of '
                         'the baseline).' + hint)


# ============================================================================ detector
def detect_spikes_janca(x, fs, *, preset='spike', band=_P, filter_order=_P, powerline=_P,
                        notch_width=_P, notch_order=_P, notch_harmonics=_P, target_fs=_P,
                        decimation=_P, window_s=_P, threshold=_P, min_distance_s=_P,
                        eps_rel=_P, return_details=False, baseline=None, combine='reference',
                        broadcast_baseline=False, gap_aware_stats=False, valid=None,
                        min_valid_fraction=None, stats_margin_s=None, channel_names=None):
    """
    Janca envelope-distribution detector (eeg_forge formulation, fixed), for spikes or, with
    ``preset='ripple'``, the ripple band.

    See the module docstring for the algorithm, the presets and the differences from the
    reference. Every parameter left at ``<preset>`` takes the value of ``preset``; any value
    given explicitly overrides it. The defaults below are those of ``preset='spike'``.

    With the defaults of the options after ``return_details`` (``baseline=None``,
    ``gap_aware_stats=False``) the detector is the original algorithm, unchanged (identical
    to eeg_forge on the parity data). The options are opt-in extensions:

    - ``baseline``: a **reference baseline** (:class:`~brainmaze_eeg.spikes.janca_baseline.JancaBaseline`,
      e.g. from a quiet or pre-injury recording) instead of / combined with the local
      background model; for channels that spike so often that the local model learns the
      spikes as background. See the README, "Reference baseline".
    - ``gap_aware_stats``: local statistics that skip gaps, for long windows on data with
      dropouts.

    Parameters
    ----------
    x : np.ndarray
        Signal, ``(n_samples,)`` or ``(n_channels, n_samples)``. Any amplitude unit
        (the detector is scale-invariant; **with a baseline the unit must be the
        baseline's**). Must be finite: NaN/inf raise ``ValueError``
        (wrap the detector in :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`
        for data with gaps).
    fs : float
        Sampling frequency of ``x`` (Hz).
    preset : {'spike', 'ripple'}
        Named parameter set (:data:`JANCA_PRESETS`). ``'spike'`` (default): the eeg_forge
        reference values. ``'ripple'``: band 80-250 Hz, ``target_fs`` 1000 Hz, everything
        else as ``'spike'``; **not validated on real ripples**.
    band : (float, float)
        Band-pass design edges in Hz (spike: 10, 60; ripple: 80, 250). Single-pass -3 dB;
        realised (zero-phase) **-6.02 dB** at both edges. Must satisfy ``0 < low < high``,
        ``high`` < Nyquist of ``fs``, and, when resampling, lose at most
        :data:`MAX_RESAMPLER_LOSS_DB` in the resampler (about ``high <= 0.85 * fs_a / 2``).
    filter_order : int
        Butterworth prototype order of the band-pass (reference: 3; 1-20).
    powerline : float or None
        Power-line frequency in Hz (reference: 50). **Set 60 for North-American data.**
        ``None`` disables the notch.
    notch_width : float
        Total width (Hz) of the band-stop, centred on ``powerline`` (reference: 5, i.e.
        +/-2.5 Hz design edges; -6 dB at the edges after zero-phase filtering). Must be
        ``< 2 * powerline``.
    notch_order : int
        Butterworth prototype order of the band-stop (reference: 3; 1-20).
    notch_harmonics : int
        Number of notches at ``k * powerline``, ``k = 1..notch_harmonics`` (reference: 1).
        Notches not fitting below Nyquist are skipped with a warning.
    target_fs : float or None
        Decimation target (Hz) (spike: 200; ripple: 1000). ``None`` keeps the input rate.
    decimation : {'integer', 'exact'}
        ``'integer'`` (default, reference): decimate by the integer factor
        ``q = floor(fs / target_fs)`` only when ``fs >= 2 * target_fs``; the analysis rate
        ``fs / q`` is then generally not ``target_fs`` (512 -> 256 Hz, 2048 -> 204.8 Hz,
        250..399 Hz -> not decimated). ``'exact'`` (ours): rational polyphase resampling
        to ``target_fs`` (within 1e-6 relative) whenever ``fs > target_fs``, so the analysis
        rate (and with it the effective time resolution of the envelope model) is the same
        for every input rate. Detections are mapped back to input samples by rounding, with
        the realised ratio.
    window_s : float
        Length (s) of the sliding window of the log-envelope statistics (reference ``w``: 5).
        Must span at least 3 analysis samples. A window longer than the analysis record issues a
        ``UserWarning``: the statistics then cover the whole (reflected) record, not a
        local window. Not used with ``combine='reference'``.
    threshold : float
        Threshold multiplier on ``mode + median`` of the local log-normal model
        (reference ``thr``: 3.65, the paper's ``k1``). Finite, > 0. With a baseline the same
        multiplier applies to the baseline's model.
    min_distance_s : float
        Minimum distance between detections in seconds (reference: 0.1). Of two close
        maxima the larger is kept. Finite, >= 0.
    eps_rel : float
        Envelope offset before the log, relative to the channel's median envelope
        (our choice, replaces the reference's absolute 1e-6; see module docstring).
        Finite, >= 0.
    return_details : bool
        Also return per-channel diagnostics (see Returns).
    baseline : JancaBaseline or None
        ``None`` (default): the original algorithm (local background model only). A
        :class:`~brainmaze_eeg.spikes.janca_baseline.JancaBaseline`: per-channel log-envelope
        statistics ``mu``, ``sd`` of a reference recording, giving the fixed threshold
        ``threshold * (exp(mu - sd**2) + exp(mu))``. Its **signal-path parameters**
        (:data:`SIGNAL_PATH_PARAMS`: band, filter orders, notch settings, target_fs,
        decimation, eps_rel) and its analysis rate ``fs_analysis`` must equal this call's
        (``ValueError`` naming every mismatch); the decision parameters (``threshold``,
        ``min_distance_s``, ``window_s``, ``combine``) are free. The amplitude unit, gain and
        montage of ``x`` must be the baseline's (not checkable; a ``UserWarning`` is issued
        when a channel's envelope level differs from the baseline's by more than
        :data:`BASELINE_LEVEL_WARN_RATIO` (x10), and when the median over channels of the
        background level (low quantile of the log-envelope, robust to dense spiking)
        differs by more than :data:`MONTAGE_LEVEL_WARN_RATIO` (x3)). The channel order is
        checked when ``channel_names`` is given and the baseline has names.
    combine : {'reference', 'min', 'max'}
        With a baseline: ``'reference'`` (default) uses the baseline's fixed threshold only;
        ``'min'`` the lower of the local and the baseline threshold at each sample (more
        sensitive: a detection above either threshold); ``'max'`` the higher (stricter:
        above both). A value other than ``'reference'`` without a baseline raises. Where the
        local threshold is undefined (``gap_aware_stats`` window coverage too low),
        ``'min'`` uses the baseline threshold (``np.fmin``: never less sensitive than
        ``'reference'``) and ``'max'`` is undefined (no detections: never more sensitive
        than the local model). With ``gap_aware_stats``, samples excluded from the
        statistics (invalid or within ``stats_margin_s`` of an invalid sample) are undefined
        in every mode.
    broadcast_baseline : bool
        Allow a **1-channel** baseline to be applied to every channel of ``x`` (default
        False: the baseline's channel count must equal ``x``'s, else ``ValueError``). A
        multichannel baseline is never broadcast.
    gap_aware_stats : bool
        ``False`` (default): the original local statistics over every sample. ``True``: the
        sliding ``mu``/``sd`` skip invalid samples (``valid``; analysis samples within
        ``stats_margin_s`` of an invalid one are skipped too) and need at least
        ``min_valid_fraction`` of each window valid; elsewhere the local threshold is NaN
        and nothing is detected. Samples excluded from the statistics are never detected
        either (threshold NaN there, whatever ``combine``): no detections on a dropout or
        on the filter transient at its edges. Computed with cumulative sums, O(n) for any
        ``window_s`` (e.g. 600-3600 s on 24 h records). Without invalid samples the result
        equals the original to rounding.
    valid : np.ndarray of bool or None
        Only with ``gap_aware_stats=True``: True on samples that may enter the statistics,
        shape of ``x``. :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` passes
        its gap mask here (filled samples are False). ``None``: every sample is valid except
        runs of exactly constant value lasting at least 0.1 s (the wrapper's
        ``flat_as_gap_s`` rule). ``x`` must still be finite.
    min_valid_fraction : float or None
        Only with ``gap_aware_stats=True``: minimum fraction (0, 1] of valid samples in a
        window (default :data:`DEFAULT_MIN_VALID_FRACTION` = 0.5).
    stats_margin_s : float or None
        Only with ``gap_aware_stats=True``: margin (s) around invalid samples excluded from
        the statistics, and from detection (default :data:`DEFAULT_STATS_MARGIN_S` = 0.5).
    channel_names : sequence of str or None
        Only with a baseline: the names of the rows of ``x``. If the baseline has
        ``channel_names`` too, they must be equal, in order (``ValueError`` otherwise, with
        a ``baseline.select(...)`` hint), except for a broadcast 1-channel baseline.

    Returns
    -------
    detections : np.ndarray or list of np.ndarray
        For 1-D input: ``int64`` array of detection **sample indices into x** (input rate),
        sorted. For 2-D input: a list with one such array per channel (row of ``x``).
        Convert to seconds with ``detections / fs``.
    details : dict or list of dict
        Only with ``return_details=True`` (one dict per channel for 2-D input):
        ``fs_analysis`` (Hz), ``up``/``down`` resampling factors (``fs_analysis = fs * up /
        down``), ``envelope`` and ``threshold`` (the threshold curve actually used, at the
        analysis rate; sample ``i`` corresponds to input sample ``i * down / up``), ``filters``
        (output of :func:`design_janca_filters`), ``preset`` and ``params`` (the resolved
        parameter values). With a baseline or ``gap_aware_stats`` also: ``threshold_local``
        (local curve, ``None`` with ``combine='reference'``), ``threshold_reference`` (the
        baseline's fixed threshold, float, or ``None``), ``combine``, ``stats_valid`` (bool
        analysis-rate mask of the samples in the statistics, or ``None``) and, with a
        baseline, ``level_ratio`` (``exp(mean log-envelope of the record - baseline mu)``;
        about 1 when the units match; dense spiking raises it) and ``background_ratio``
        (the same at the :data:`MONTAGE_LEVEL_QUANTILE` quantile: record vs the baseline
        model ``mu + ndtri(q) * sd``; about 1 when gain and montage match, little moved by
        spikes). Both are ``None`` for an all-zero envelope.

    Raises
    ------
    ValueError, TypeError
        Invalid parameters (see :func:`janca_params`), a band edge at/above the Nyquist
        frequency or too close to it after resampling, a window shorter than 3 analysis
        samples, a record too short for the zero-phase filters, a 2-D array with more rows
        than columns (probably transposed), a non-finite value (NaN/inf) in ``x``, a
        baseline whose signal path or channel count does not match, or an option that would
        be ignored (``combine``/``broadcast_baseline`` without a baseline, ``valid``,
        ``min_valid_fraction``, ``stats_margin_s`` without ``gap_aware_stats``).
    """
    p = janca_params(preset, band=band, filter_order=filter_order, powerline=powerline,
                     notch_width=notch_width, notch_order=notch_order,
                     notch_harmonics=notch_harmonics, target_fs=target_fs,
                     decimation=decimation, window_s=window_s, threshold=threshold,
                     min_distance_s=min_distance_s, eps_rel=eps_rel)
    combine, min_valid_fraction, stats_margin_s = _resolve_options(
        baseline, combine, broadcast_baseline, gap_aware_stats, valid, min_valid_fraction,
        stats_margin_s)
    fs = chk.number('fs', fs, gt=0)
    X, one_d = _as_channels_first(x)
    n = X.shape[1]

    filters, up, down, fs_a, padlen = _signal_path(p, fs)
    win = int(p['window_s'] * fs_a)
    if win < 3:
        raise ValueError(f"window_s={p['window_s']} s spans {win} analysis sample(s) at "
                         f'{fs_a:g} Hz; it must span at least 3')
    if win % 2 == 0:
        win += 1
    dist = max(int(p['min_distance_s'] * fs_a), 1)
    if n <= padlen:
        raise ValueError(f'record too short: {n} samples; the zero-phase filters need more '
                         f'than {padlen} samples ({padlen / fs:g} s at fs={fs:g} Hz)')
    n_a = -(-n * up // down)                      # analysis length (resample_poly output)
    uses_local = baseline is None or combine != 'reference'
    if win > n_a and uses_local:
        warnings.warn(f"window_s={p['window_s']:g} s ({win} analysis samples) is longer than "
                      f'the record ({n_a} samples = {n_a / fs_a:.3g} s at {fs_a:g} Hz): the '
                      'background statistics then cover the whole record (reflected at both '
                      'ends) instead of a local window, so detections are relative to the '
                      'whole-record background. Use a shorter window_s or a longer record.',
                      UserWarning, stacklevel=2)

    _check_finite(X, 'x')
    if baseline is not None:
        baseline.check_compatible(p, fs_a)
        mu_b, sd_b = _baseline_rows(baseline, X.shape[0], broadcast_baseline)
        names = _names_tuple(channel_names, X.shape[0])
        if names is not None:
            _check_channel_names(baseline, names, broadcast_baseline)
    elif channel_names is not None:
        raise ValueError('channel_names only apply with a baseline (they are checked against '
                         "the baseline's channel_names)")
    V = _stats_valid_mask(valid, X, fs, one_d) if gap_aware_stats else None
    margin_a = int(np.ceil(stats_margin_s * fs_a - 1e-9))
    extended = baseline is not None or gap_aware_stats

    results, details, off_level, bg_ratios = [], [], [], []
    for c in range(X.shape[0]):
        # -- preprocessing, one channel at a time (bounded memory) ------------------------
        e = _channel_envelope(X[c], filters, up, down)
        ok = _analysis_ok(V[c], up, down, e.size, margin_a) if V is not None else None

        # -- threshold: local model (reference), baseline, or a combination ----------------
        thr_local = (_local_threshold(e, p['threshold'], p['eps_rel'], win, ok,
                                      min_valid_fraction) if uses_local else None)
        t_ref, level_ratio, bg_ratio = None, None, None
        if baseline is None:
            thr_curve = thr_local
        else:
            t_ref = float(janca_threshold(mu_b[c], sd_b[c], p['threshold']))
            if combine == 'reference':
                thr_curve = np.full(e.shape, t_ref)
            elif combine == 'min':
                # np.fmin: where the local threshold is undefined (gap_aware_stats coverage)
                # the reference applies, so 'min' is never less sensitive than 'reference'
                thr_curve = np.fmin(thr_local, t_ref)
            else:
                thr_curve = np.maximum(thr_local, t_ref)      # NaN (undefined) propagates
            ev = e if ok is None else e[ok]
            scale = _envelope_scale(ev)
            if scale > 0:
                log_ev = np.log(ev + p['eps_rel'] * scale)
                level_ratio = float(np.exp(np.mean(log_ev) - mu_b[c]))
                if not (1 / BASELINE_LEVEL_WARN_RATIO <= level_ratio
                        <= BASELINE_LEVEL_WARN_RATIO):
                    off_level.append((c, level_ratio))
                q = MONTAGE_LEVEL_QUANTILE
                bg_ratio = float(np.exp(np.quantile(log_ev, q)
                                        - (mu_b[c] + ndtri(q) * sd_b[c])))
                bg_ratios.append(bg_ratio)
        if ok is not None:
            # gap_aware_stats: samples excluded from the statistics (invalid, or within
            # stats_margin_s of an invalid sample) are undefined for every combine mode:
            # never a detection on a dropout or on the transient at its edges
            thr_curve = np.where(ok, thr_curve, np.nan)

        # -- detections: envelope maxima above the threshold (none where it is NaN) -------
        idx = np.zeros(0, dtype=np.int64)
        undefined = np.isnan(thr_curve)
        if not undefined.all():
            height = np.where(undefined, np.inf, thr_curve) if undefined.any() else thr_curve
            pk = find_peaks(e, height=height, distance=dist)[0]
            if up == 1:
                idx = pk.astype(np.int64) * down
            else:
                idx = np.minimum(np.round(pk * (down / up)).astype(np.int64), n - 1)
        results.append(idx)
        if return_details:
            d = {'fs_analysis': fs_a, 'up': up, 'down': down, 'envelope': e,
                 'threshold': thr_curve, 'filters': filters, 'preset': preset,
                 'params': dict(p)}
            if extended:
                d.update({'threshold_local': thr_local, 'threshold_reference': t_ref,
                          'combine': combine if baseline is not None else None,
                          'stats_valid': ok, 'level_ratio': level_ratio,
                          'background_ratio': bg_ratio})
            details.append(d)

    if off_level:
        warnings.warn(
            'the envelope level of channel(s) '
            + ', '.join(f'{c} (x{r:.3g})' for c, r in off_level)
            + f' differs from the baseline\'s by more than x{BASELINE_LEVEL_WARN_RATIO:g}. '
            'Check that x has the unit, gain and montage of the baseline recording (e.g. uV '
            'vs V) and that the channels are in the same order; a mismatched baseline gives '
            'a meaningless threshold.', UserWarning, stacklevel=2)
    if bg_ratios:
        med = float(np.median(bg_ratios))
        lo, hi = 1 / MONTAGE_LEVEL_WARN_RATIO, MONTAGE_LEVEL_WARN_RATIO
        if not lo <= med <= hi:
            warnings.warn(
                f'the background envelope level of x is x{med:.3g} the baseline\'s (median '
                f'over {len(bg_ratios)} channel(s) of the level at the '
                f'{MONTAGE_LEVEL_QUANTILE:g} quantile of the log-envelope; expected '
                f'x{lo:.3g}-x{hi:g}). Check the gain, unit and montage/reference of x against '
                'the baseline recording; a mismatched baseline gives a meaningless threshold.',
                UserWarning, stacklevel=2)
    if one_d:
        results = results[0]
        details = details[0] if details else details
    return (results, details) if return_details else results


def _check_finite(X, name):
    bad = ~np.isfinite(X)
    if bad.any():
        ch = np.flatnonzero(bad.any(axis=-1)).tolist()
        raise ValueError(f"'{name}' contains NaN/inf (channels {ch}). The raw detectors need "
                         'finite input; use brainmaze_eeg.spikes.GapAwareSpikeDetector to fill '
                         'gaps and drop detections near them.')


class JancaDetector:
    """
    :func:`detect_spikes_janca` as a detector object (the protocol of
    :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`).

    ``JancaDetector(preset, **overrides).detect(x, fs)`` equals
    ``detect_spikes_janca(x, fs, preset=preset, **overrides)`` for 2-D ``x``
    ``(n_channels, n_samples)``: a list with one ``int64`` array of sample indices per
    channel. All parameters are resolved and validated at construction
    (:func:`janca_params`: names, types, finiteness, ranges; a baseline's signal path is
    checked against them); the checks that need the sampling rate (Nyquist, resampler loss,
    window and record length, the baseline's ``fs_analysis``) run in :meth:`detect`.

    ``baseline``, ``combine``, ``broadcast_baseline``, ``gap_aware_stats``,
    ``min_valid_fraction`` and ``stats_margin_s`` are the options of
    :func:`detect_spikes_janca` (defaults: the original algorithm). Inside
    :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`:

    - with ``gap_aware_stats=True`` the detector receives the wrapper's gap mask
      (``accepts_valid``), so filled samples never enter the local statistics;
    - with a baseline it receives the caller's channel indices (``accepts_channels``), so each
      channel is compared with its own baseline row even when the wrapper feeds channels one
      at a time or leaves all-missing channels out.

    ``channel_names`` (only with a baseline): names of the **full montage** given to
    :meth:`detect` (or to the wrapper). If the baseline has names too they must be equal, in
    order (checked here, ``ValueError``); :meth:`detect` passes the names of the rows it
    receives to :func:`detect_spikes_janca`.

    Examples::

        JancaDetector()                                  # spikes, eeg_forge values
        JancaDetector(powerline=60)                      # spikes, North-American mains
        JancaDetector('ripple')                          # 80-250 Hz at ~1 kHz (unvalidated)
        JancaDetector('ripple', notch_harmonics=5, threshold=4.0)
        JancaDetector(powerline=60, baseline=JancaBaseline.load('pre_injury.json'))
        JancaDetector(window_s=1800, gap_aware_stats=True)
    """

    output = 'indices'
    channel_independent = True     # channels never interact: the wrapper may feed them singly

    def __init__(self, preset='spike', *, baseline=None, combine='reference',
                 broadcast_baseline=False, gap_aware_stats=False, min_valid_fraction=None,
                 stats_margin_s=None, channel_names=None, **overrides):
        if 'return_details' in overrides:
            raise TypeError('return_details is not a detector parameter; call '
                            'detect_spikes_janca directly for details')
        if 'valid' in overrides:
            raise TypeError('valid is a per-call argument of detect(), not a detector parameter')
        self.params = janca_params(preset, **overrides)
        self.preset = preset
        self.overrides = dict(overrides)
        _resolve_options(baseline, combine, broadcast_baseline, gap_aware_stats, None,
                         min_valid_fraction, stats_margin_s)
        if baseline is not None:
            baseline.check_compatible(self.params)
        names = _names_tuple(channel_names)
        if names is not None:
            if baseline is None:
                raise ValueError('channel_names only apply with a baseline (they are checked '
                                 "against the baseline's channel_names)")
            if not (baseline.n_channels == 1 and broadcast_baseline):
                if len(names) != baseline.n_channels:
                    raise ValueError(f'channel_names has {len(names)} name(s), the baseline '
                                     f'{baseline.n_channels} channel(s)')
                _check_channel_names(baseline, names, broadcast_baseline)
        self.channel_names = names
        self.baseline = baseline
        self.combine = combine
        self.broadcast_baseline = bool(broadcast_baseline)
        self.gap_aware_stats = bool(gap_aware_stats)
        self.min_valid_fraction = min_valid_fraction
        self.stats_margin_s = stats_margin_s

    @property
    def accepts_valid(self):
        """True with ``gap_aware_stats``: the wrapper then passes its gap mask as ``valid``."""
        return self.gap_aware_stats

    @property
    def accepts_channels(self):
        """True with a baseline: the wrapper then passes the caller's channel indices."""
        return self.baseline is not None

    def _options(self):
        opts = {}
        if self.baseline is not None:
            opts.update(baseline=self.baseline, combine=self.combine,
                        broadcast_baseline=self.broadcast_baseline)
            if self.channel_names is not None:
                opts.update(channel_names=self.channel_names)
        if self.gap_aware_stats:
            opts.update(gap_aware_stats=True, min_valid_fraction=self.min_valid_fraction,
                        stats_margin_s=self.stats_margin_s)
        return opts

    def __repr__(self):
        args = [repr(self.preset)] + [f'{k}={v!r}' for k, v in self.overrides.items()]
        args += [f'{k}={v!r}' for k, v in self._options().items()
                 if not (k in ('min_valid_fraction', 'stats_margin_s') and v is None)]
        return f'JancaDetector({", ".join(args)})'

    def detect(self, x, fs, valid=None, channels=None, n_channels=None):
        """
        Detection sample indices per channel of ``x`` ``(n_channels, n_samples)``.

        ``valid`` (with ``gap_aware_stats=True`` only): boolean mask of ``x``'s shape, True on
        samples that may enter the local statistics. ``channels`` / ``n_channels`` (used by
        :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` when there is a
        baseline): the rows of ``x`` are channels ``channels`` (integer indices) of a montage
        of ``n_channels``; the baseline must have ``n_channels`` rows (or 1 with
        ``broadcast_baseline=True``) and row ``channels[i]`` is used for ``x[i]``.
        ``channels`` may instead be channel **names** (str) of a baseline with
        ``channel_names`` (``n_channels`` is then not needed). Anything else (floats, bools)
        raises ``TypeError``.
        """
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f'JancaDetector.detect expects (n_channels, n_samples), got {x.shape}')
        opts = self._options()
        if valid is not None:
            if not self.gap_aware_stats:
                raise ValueError('valid is only used with gap_aware_stats=True')
            opts['valid'] = valid
        if channels is not None and self.baseline is not None:
            channels = [channels] if isinstance(channels, (str, int, np.integer)) else list(
                np.asarray(channels, dtype=object).reshape(-1))
            if len(channels) != x.shape[0]:
                raise ValueError(f'channels has {len(channels)} entries for {x.shape[0]} rows')
            if channels and all(isinstance(c, str) for c in channels):
                return self._detect_named(x, fs, channels, opts)
            if not all(isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_))
                       for c in channels):
                raise TypeError(f'channels must be integer indices (or channel names), got '
                                f'{channels!r}')
            channels = np.asarray(channels, dtype=np.int64)
            if n_channels is None:
                raise TypeError('n_channels (the size of the full montage) is required with '
                                'channels')
            n_channels = chk.integer('n_channels', n_channels, ge=1)
            if channels.size and (channels.min() < 0 or channels.max() >= n_channels):
                raise ValueError(f'channels {channels.tolist()} out of range for a montage of '
                                 f'{n_channels}')
            nb = self.baseline.n_channels
            if nb == n_channels:
                opts['baseline'] = self.baseline.select(channels)
            elif not (nb == 1 and self.broadcast_baseline):
                _baseline_rows(self.baseline, n_channels, self.broadcast_baseline)  # raises
            if self.channel_names is not None:
                if len(self.channel_names) != n_channels:
                    raise ValueError(f'channel_names has {len(self.channel_names)} name(s) '
                                     f'for a montage of {n_channels}')
                opts['channel_names'] = [self.channel_names[k] for k in channels]
        return detect_spikes_janca(x, fs, preset=self.preset, **self.params, **opts)

    def _detect_named(self, x, fs, names, opts):
        """:meth:`detect` with ``channels`` given as names of the baseline's channels."""
        if self.channel_names is not None and not set(names) <= set(self.channel_names):
            raise ValueError(f'channels {names} are not all in channel_names '
                             f'{list(self.channel_names)}')
        if not (self.baseline.n_channels == 1 and self.broadcast_baseline):
            if self.baseline.channel_names is None:
                raise ValueError('channels given as names need a baseline with channel_names')
            opts['baseline'] = self.baseline.select(names)        # KeyError if unknown
        opts['channel_names'] = list(names)
        return detect_spikes_janca(x, fs, preset=self.preset, **self.params, **opts)


# ===================================================================== MATLAB-v24 port
class SpikeDetectorHilbert:
    """
    Janca envelope-distribution IED detector, port of MATLAB ``spike_detector_hilbert_v24``.

    Use this when you need v24's full output (detection CDF/PDF weights, multichannel
    discharge grouping, the ambiguous ``k2`` class) or comparability with MATLAB v24
    results; otherwise prefer :func:`detect_spikes_janca`. **Input layout is
    (n_samples, n_channels)** (MATLAB convention), the opposite of the rest of the package.

    Pipeline (v24): resample to ``decimation`` Hz -> power-line notch comb -> 1 Hz
    high-pass (Butterworth order 2) -> per segment: band-pass ``bandwidth`` -> Hilbert
    envelope -> per-window (``winsize``/``noverlap``) log-normal MLE, smoothed and
    interpolated -> threshold ``k1*(mode+median) - k3*(mean-mode)`` -> local maxima,
    poly-spike union -> output.

    Filters (all zero-phase, ``sos``; verified by the test-suite):

    - **Power-line notch comb**: 2nd-order IIR notches at ``main_hum_freq`` and harmonics
      up to ``1.1 * bandwidth[1]``, unit gain away from the notch. v24 uses a fixed pole
      radius 0.985, which gives a ~1 Hz wide notch at 200 Hz but widens proportionally
      with the rate (~24 Hz at 5 kHz with ``decimation=0``). Here the radius is
      ``1 - 0.015 * 200 / fs`` so the width stays ~1 Hz (identical to v24 at 200 Hz).
    - **High-pass** 1 Hz Butterworth order 2.
    - **Band-pass** ``f_type``:

      1. Chebyshev-II (default): minimum-order low-pass and high-pass meeting
         ``cheb_rp`` dB (6) max loss at the band edges and ``cheb_rs`` dB (60) stop-band
         attenuation ``cheb_transition_hz`` = (5, 10) Hz beyond the low/high edge, i.e. the
         v24 spec (normalised 0.05/0.1 at 200 Hz) expressed in Hz so it is valid at any
         rate. **Effective zero-phase response: about -12 dB at the band edges (v24's 6 dB
         single-pass spec), < -120 dB in the stop bands.**
         *Fixed defect:* the earlier port passed the **pass-band** edge as ``cheby2``'s
         ``Wn`` (which is the **stop-band** edge) and discarded the order-design ``Wn``,
         shrinking the effective band to ~18-48 Hz (15 Hz attenuated to 0.03, 50 Hz to
         0.15 of the input power).
      2. Butterworth order 4 high-pass + order 4 low-pass (-6 dB at the edges).
      3. FIR (``firwin``, ``fs/2`` taps, odd) high-pass + low-pass (-12 dB at the edges).

      **Change from v24:** v24 replaced Chebyshev by Butterworth (with a warning) whenever
      ``decimation`` was not 200 Hz, because its Chebyshev spec was normalised to 200 Hz.
      Here the spec is in Hz and valid at any analysis rate (verified at 200 Hz-32 kHz), so
      ``f_type`` is always used as given. Pass ``f_type=2`` to reproduce v24 at other rates.

    Resampling (``decimation`` > 0 and different from ``fs``): rational polyphase
    (:func:`scipy.signal.resample_poly`) by the smallest-denominator ratio ``up / down``
    within 1e-6 (relative) of ``decimation / fs``, so any real input rate works (TDT
    24414.0625 Hz, 511.99 Hz). The analysis then runs at the realised rate
    ``fs * up / down`` (filters designed and positions converted at that rate, so there is
    no timing drift). The band must lie below the input Nyquist as well, and the resampler's
    anti-alias filter may lose at most :data:`MAX_RESAMPLER_LOSS_DB` at ``bandwidth[1]``.

    Parameters (defaults follow v24)
    --------------------------------
    bandwidth : (float, float)
        Band-pass edges [low, high] in Hz. Default [10, 60].
    k1 : float
        Threshold multiplier for obvious spikes. Default 3.65.
    k2 : float
        Threshold multiplier for ambiguous spikes, ``0 < k2 <= k1`` (MATLAB v24 help:
        "k1 >= k2"). A local maximum above the ``k2`` threshold but not above ``k1`` is
        reported as ambiguous (``con`` 0.5) only if an obvious detection on **any**
        channel lies within the preceding 10 ms (``[i - 10 ms, i]``; v24 tests the single
        sample ``i - 10 ms``, which we read as a typo for this window). Default equals
        ``k1`` (ambiguous class disabled). With ``k2 < k1`` the result of a channel depends
        on the other channels, so :attr:`channel_independent` is False and
        :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` passes the whole
        montage. (An earlier version enforced ``k2 >= k1``, which inverted the v24
        constraint; the ambiguous class could then never fire.)
        **Not comparable to MATLAB v24** when ``k2 < k1``: v23/v25 test the literal single
        sample, and on real 15-channel iEEG the window used here gives 4-60x more ambiguous
        detections (a symmetric +-10 ms window 12-50 % more still). See the README
        ("Ambiguous class is not comparable to v24"). Defaults are unaffected.
    k3 : float
        Threshold tilt term. Default 0.
    main_hum_freq : float or None
        Mains frequency to notch (Hz). Default 50 (**use 60 for North-American data**).
        ``None`` disables the notch.
    decimation : float
        Target analysis sampling rate (Hz). Default 200. Set 0 to keep the input rate.
    buffering : float
        Core window length for batch processing (seconds). Default 300.
    winsize, noverlap : float
        Envelope-model window and overlap in **seconds**. Defaults 5 and 4. A ``winsize`` longer
        than the analysis record issues a ``UserWarning`` (one fit for the whole record).
    polyspike_union_time : float
        Poly-spike union interval (seconds). Default 0.12.
    discharge_tol : float
        Grouping tolerance for multichannel events (seconds). Default 0.005.
    f_type : int
        Band-pass family: 1 = Chebyshev-II (default), 2 = Butterworth, 3 = FIR.
    cheb_rp, cheb_rs : float
        Chebyshev-II max pass-band loss / min stop-band attenuation, single pass, dB.
        Defaults 6 and 60 (v24).
    cheb_transition_hz : (float, float)
        Chebyshev-II transition widths (Hz) below the low and above the high edge.
        Default (5, 10) (= v24 at 200 Hz).
    beta : float
        Low edge (Hz) of beta rejection. ``inf`` (default) disables it; any finite value
        raises ``NotImplementedError``.

    Input must be finite (NaN/inf raise ``ValueError``). For data with gaps use
    ``GapAwareSpikeDetector(SpikeDetectorHilbert(...))``: it calls :meth:`detect`, which
    takes ``(n_channels, n_samples)`` and returns detection sample indices per channel.

    Buffering scheme
    ----------------
    The record is partitioned into contiguous *core* windows that tile ``[0, N)`` exactly.
    Each core is analysed inside a block extended by ``margin = 3 * winsize`` samples on each
    side, and only detections inside the core are kept, so every detection is produced
    exactly once (verified by a whole-vs-buffered test).

    Not implemented: beta/mu-activity rejection and the ``ti_switch == 2`` timing mode.
    """

    _CHEB_HUM_R_AT_200 = 0.985

    def __init__(self, **kwargs):
        self.bandwidth = [10.0, 60.0]
        self.k1 = 3.65
        self.k2 = self.k1
        self.k3 = 0.0
        self.main_hum_freq = 50.0
        self.decimation = 200.0
        self.buffering = 300.0
        self.winsize = 5.0          # seconds
        self.noverlap = 4.0         # seconds
        self.polyspike_union_time = 0.12
        self.discharge_tol = 0.005
        self.f_type = 1
        self.cheb_rp = 6.0
        self.cheb_rs = 60.0
        self.cheb_transition_hz = (5.0, 10.0)
        self.beta = np.inf
        for key, value in kwargs.items():
            if (key.startswith('_') or not hasattr(self, key) or callable(getattr(self, key))
                    or isinstance(getattr(type(self), key, None), property)):
                raise TypeError(f'unknown parameter {key!r}')
            setattr(self, key, value)
        if 'k2' not in kwargs:
            self.k2 = self.k1
        self._validate()

    def _validate(self):
        """Check every parameter (type, finiteness, range); run at construction and by
        :meth:`run`, so attributes changed after construction are checked too."""
        self.bandwidth = list(chk.pair('bandwidth', self.bandwidth, gt=0))
        self.k1 = chk.number('k1', self.k1, gt=0)
        self.k2 = chk.number('k2', self.k2, gt=0)
        if self.k2 > self.k1:
            raise ValueError(f'k2 ({self.k2}) must be <= k1 ({self.k1}): k2 is the lower, '
                             'ambiguous-spike threshold (MATLAB v24: "k1 >= k2"); '
                             'k2 = k1 disables the ambiguous class')
        self.k3 = chk.number('k3', self.k3)
        self.main_hum_freq = chk.number('main_hum_freq', self.main_hum_freq, gt=0,
                                        allow_none=True)
        self.decimation = chk.number('decimation', self.decimation, ge=0, allow_none=True)
        self.buffering = chk.number('buffering', self.buffering, gt=0)
        self.winsize = chk.number('winsize', self.winsize, gt=0)
        self.noverlap = chk.number('noverlap', self.noverlap, ge=0)
        if not self.noverlap < self.winsize:
            raise ValueError(f'noverlap ({self.noverlap} s) must be < winsize ({self.winsize} s)')
        self.polyspike_union_time = chk.number('polyspike_union_time',
                                               self.polyspike_union_time, ge=0)
        self.discharge_tol = chk.number('discharge_tol', self.discharge_tol, ge=0)
        self.f_type = chk.integer('f_type', self.f_type)
        if self.f_type not in (1, 2, 3):
            raise ValueError(f'f_type must be 1 (Chebyshev-II), 2 (Butterworth) or 3 (FIR), '
                             f'got {self.f_type!r}')
        self.cheb_rp = chk.number('cheb_rp', self.cheb_rp, gt=0)
        self.cheb_rs = chk.number('cheb_rs', self.cheb_rs, gt=self.cheb_rp)
        t_lo, t_hi = self.cheb_transition_hz if np.ndim(self.cheb_transition_hz) == 1 and \
            len(self.cheb_transition_hz) == 2 else (None, None)
        self.cheb_transition_hz = (chk.number('cheb_transition_hz[0]', t_lo, gt=0),
                                   chk.number('cheb_transition_hz[1]', t_hi, gt=0))
        self.beta = chk.number('beta', self.beta, allow_inf=True)
        if np.isfinite(self.beta):
            raise NotImplementedError('beta/mu rejection is not implemented (beta must be inf)')

    # ------------------------------------------------------------------ filter design
    def design_filters(self, fs):
        """
        Design the filters applied at the analysis rate ``fs`` (i.e. after decimation).

        Returns
        -------
        dict
            ``'notches'``: list of ``(centre_hz, sos)``; ``'highpass'``: 1 Hz ``sos``;
            ``'bandpass'``: list of filters applied in sequence, each ``sos`` (IIR) or
            ``('fir', taps)``; ``'f_type'``: family actually used.
        """
        fs = float(fs)
        lo, hi = flt.check_band(self.bandwidth, fs, name='bandwidth')
        out = {'notches': [], 'highpass': flt.butter_highpass(1.0, fs, 2)}
        if self.main_hum_freq is not None:
            r = 1.0 - (1.0 - self._CHEB_HUM_R_AT_200) * 200.0 / fs
            f = float(self.main_hum_freq)
            while f <= 1.1 * hi and f < fs / 2:
                out['notches'].append((f, flt.iir_notch_sos(f, fs, r)))
                f += self.main_hum_freq
        ftype = self.f_type
        out['f_type'] = ftype
        if ftype == 1:
            t_lo, t_hi = self.cheb_transition_hz
            if lo - t_lo <= 0 or hi + t_hi >= fs / 2:
                raise ValueError(
                    f'Chebyshev-II band-pass needs bandwidth[0] - {t_lo} Hz > 0 and '
                    f'bandwidth[1] + {t_hi} Hz < Nyquist ({fs / 2} Hz); got {self.bandwidth}. '
                    'Adjust cheb_transition_hz or use f_type=2 (Butterworth).')
            out['bandpass'] = [
                flt.cheby2_highpass(lo, lo - t_lo, fs, self.cheb_rp, self.cheb_rs),
                flt.cheby2_lowpass(hi, hi + t_hi, fs, self.cheb_rp, self.cheb_rs)]
        elif ftype == 2:
            out['bandpass'] = [flt.butter_highpass(lo, fs, 4), flt.butter_lowpass(hi, fs, 4)]
        else:
            n = int(fs / 2) | 1
            out['bandpass'] = [('fir', firwin(n, lo, pass_zero='highpass', fs=fs)),
                               ('fir', firwin(n, hi, fs=fs))]
        return out

    @staticmethod
    def _apply(filt, d):
        if isinstance(filt, tuple) and filt[0] == 'fir':
            return filtfilt(filt[1], 1.0, d, axis=0)
        return sosfiltfilt(filt, d, axis=0)

    # ------------------------------------------------------------------ public
    def run(self, d, fs):
        """
        Detect IEDs in ``d`` sampled at ``fs``.

        Parameters
        ----------
        d : np.ndarray
            Signal, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
        fs : float
            Input sampling frequency (Hz).

        Returns
        -------
        out : dict
            Per-detection arrays: ``pos`` (s), ``dur`` (s), ``chan`` (int),
            ``con`` (1 obvious / 0.5 ambiguous), ``weight`` (envelope CDF), ``pdf``.
        discharges : dict
            Per multichannel event: ``MV`` [n_events, n_chan] spike type, ``MA`` max envelope
            above background, ``MP`` start position (s), ``MD`` duration (s), ``MW`` CDF weight,
            ``MPDF`` pdf.
        d_decim : np.ndarray
            Decimated, hum-notched, 1 Hz high-passed signal ``(n_samples_dec, n_chan)``.
        envelope : np.ndarray
            Hilbert envelope of the band-passed signal ``(n_samples_dec, n_chan)``.
        background : np.ndarray
            Threshold curves ``(n_samples_dec, n_chan, 2)`` for k1 and k2.
        envelope_pdf : np.ndarray
            Log-normal PDF of the envelope ``(n_samples_dec, n_chan)``.
        """
        self._validate()
        fs = chk.number('fs', fs, gt=0)

        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            d = d[:, None]
        elif d.ndim != 2:
            raise ValueError("'d' must be 1-D or 2-D (n_samples, n_channels)")
        if d.shape[1] > d.shape[0]:
            raise ValueError(f"'d' has shape {d.shape}: more channels than samples. "
                             'SpikeDetectorHilbert expects (n_samples, n_channels); transpose '
                             'your array (d.T).')

        up, down = self._ratio(fs)
        fsd = fs * up / down                      # realised analysis rate
        hi = self.bandwidth[1]
        if hi >= fs / 2:
            raise ValueError(f'bandwidth high edge ({hi} Hz) must be < the Nyquist frequency of '
                             f'the input ({fs / 2:g} Hz); upsampling cannot add content above it')
        if hi >= fsd / 2:
            raise ValueError(f'bandwidth high edge ({hi} Hz) must be < the Nyquist frequency of '
                             f'the analysis rate ({fsd / 2:g} Hz); raise decimation')
        if (up, down) != (1, 1):
            _check_resampler_edge(hi, fs, up, down, fsd, name='bandwidth')
        filters = self.design_filters(fsd)

        _check_finite(d.T, 'd')

        # -- decimate (per channel), then notch mains + 1 Hz high-pass -----------
        d_dec = resample_poly(d, up, down, axis=0) if (up, down) != (1, 1) else d.copy()
        need = max([_sos_padlen(filters['highpass'])]
                   + [_sos_padlen(s) for _, s in filters['notches']]
                   + [3 * f[1].size if isinstance(f, tuple) else _sos_padlen(f)
                      for f in filters['bandpass']])
        if d_dec.shape[0] <= need:
            raise ValueError(f'record too short: {d_dec.shape[0]} samples at the analysis rate '
                             f'{fsd:g} Hz; the zero-phase filters need more than {need}')
        for _, sos in filters['notches']:
            d_dec = sosfiltfilt(sos, d_dec, axis=0)
        d_decim = sosfiltfilt(filters['highpass'], d_dec, axis=0)

        n = d_decim.shape[0]
        winsize = int(round(self.winsize * fsd))
        if winsize > n:
            warnings.warn(f'winsize={self.winsize:g} s ({winsize} samples) is longer than the '
                          f'record ({n} samples = {n / fsd:.3g} s at {fsd:g} Hz): one '
                          'log-normal fit covers the whole record instead of sliding '
                          'windows. Use a shorter winsize or a longer record.',
                          UserWarning, stacklevel=2)
        margin = 3 * winsize
        core = max(int(round(self.buffering * fsd)), winsize)

        # -- core-partition buffering (overlap-invariant) ------------------------
        markers_high = np.zeros((n, d_decim.shape[1]), dtype=bool)
        markers_low = np.zeros((n, d_decim.shape[1]), dtype=bool)
        envelope = np.zeros_like(d_decim)
        background = np.zeros((n, d_decim.shape[1], 2))
        envelope_cdf = np.zeros_like(d_decim)
        envelope_pdf = np.zeros_like(d_decim)

        starts = list(range(0, n, core)) if n > core else [0]
        for a in starts:
            b = min(a + core, n)
            bs = max(a - margin, 0)
            be = min(b + margin, n)
            blk = d_decim[bs:be]
            env, mh, ml, bg, cdf, pdf = self._detect_block(blk, fsd, winsize, filters)

            # write back the core slice [a, b) from block-local coords
            lo, hi = a - bs, b - bs
            envelope[a:b] = env[lo:hi]
            background[a:b] = bg[lo:hi]
            envelope_cdf[a:b] = cdf[lo:hi]
            envelope_pdf[a:b] = pdf[lo:hi]
            markers_high[a:b] = mh[lo:hi]
            markers_low[a:b] = ml[lo:hi]

        # blank first/last second (filter transient), per the reference
        edge = int(round(fsd))
        if n > 2 * edge:
            markers_high[:edge] = markers_high[-edge:] = False
            markers_low[:edge] = markers_low[-edge:] = False

        out = self._markers_to_out(markers_high, markers_low, envelope_cdf, envelope_pdf, fsd)
        discharges = self._group_discharges(out, envelope, background, envelope_cdf,
                                            envelope_pdf, d_decim, fsd)
        return out, discharges, d_decim, envelope, background, envelope_pdf

    output = 'indices'

    @property
    def channel_independent(self):
        """True unless the ambiguous class is enabled (``k2 < k1``): ambiguous detections
        are accepted only next to an obvious detection on *any* channel."""
        return self.k2 == self.k1

    def detect(self, x, fs):
        """
        Detector protocol of :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`.

        Parameters
        ----------
        x : np.ndarray, shape (n_channels, n_samples)
            **Channels first** (unlike :meth:`run`).
        fs : float

        Returns
        -------
        list of np.ndarray
            Per channel, sorted ``int64`` sample indices (input rate) of the detections
            (obvious and, with ``k2 < k1``, ambiguous ones), ``round(pos * fs)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f'detect expects (n_channels, n_samples), got {x.shape}')
        out = self.run(x.T, fs)[0]
        pos = np.asarray(out['pos'], dtype=float)
        chan = np.asarray(out['chan']).astype(int) if pos.size else np.zeros(0, int)
        n = x.shape[1]
        return [np.unique(np.minimum(np.round(pos[chan == c] * fs).astype(np.int64), n - 1))
                for c in range(x.shape[0])]

    # ------------------------------------------------------------- core science
    def _detect_block(self, d, fs, winsize, filters):
        """Band-pass, envelope, threshold and mark one block of shape (n_samples, n_chan)."""
        d_bp = d
        for filt in filters['bandpass']:
            d_bp = self._apply(filt, d_bp)
        n, nch = d_bp.shape
        envelope = np.zeros((n, nch))
        markers_high = np.zeros((n, nch), dtype=bool)
        markers_low = np.zeros((n, nch), dtype=bool)
        background = np.zeros((n, nch, 2))
        cdf = np.zeros((n, nch))
        pdf = np.zeros((n, nch))

        step = max(winsize - int(round(self.noverlap * fs)), 1)
        index = np.arange(0, max(n - winsize + 1, 1), step)

        for ch in range(nch):
            if not d_bp[:, ch].any():
                continue
            (envelope[:, ch], markers_high[:, ch], markers_low[:, ch],
             background[:, ch, :], cdf[:, ch], pdf[:, ch]) = self._one_channel(
                d_bp[:, ch], fs, index, winsize)
        return envelope, markers_high, markers_low, background, cdf, pdf

    def _one_channel(self, d, fs, index, winsize):
        envelope = np.abs(hilbert(d))
        k1, k2, k3 = self.k1, self.k2, self.k3

        # per-window MLE of log-normal params on the envelope
        phat = np.zeros((len(index), 2))
        for k, i0 in enumerate(index):
            seg = envelope[i0:i0 + winsize]
            seg = seg[seg > 0]
            if seg.size == 0:
                phat[k] = [0.0, 1.0]
            else:
                logs = np.log(seg)
                phat[k] = [logs.mean(), logs.std()]

        r = envelope.shape[0] / max(len(index), 1)
        n_avg = int(round((winsize / fs) * (fs / r))) if r > 0 else 1
        # smooth the per-window params; filtfilt needs len(phat) > padlen (= 3*(n_avg-1))
        if n_avg > 1 and phat.shape[0] > 3 * (n_avg - 1) + 1:
            phat = filtfilt(np.ones(n_avg) / n_avg, 1, phat, axis=0)

        # interpolate window params to a full-length threshold "background" curve
        if phat.shape[0] > 1:
            centers = index + round(winsize / 2)
            xs = np.arange(centers[0], centers[-1] + 1)
            # cubic interpolation needs >= 4 points; short recordings (2-3 window centers)
            # fall back to linear rather than raising a ValueError inside interp1d.
            kind = 'cubic' if centers.size >= 4 else 'linear'
            pi = np.empty((xs.size, 2))
            for c in range(2):
                pi[:, c] = interp1d(centers, phat[:, c], kind=kind,
                                    fill_value='extrapolate')(xs)
            top = int(np.floor(winsize / 2))
            head = np.repeat(pi[:1], top, axis=0)
            tail = np.repeat(pi[-1:], max(envelope.shape[0] - pi.shape[0] - top, 0), axis=0)
            phat_int = np.vstack([head, pi, tail])[:envelope.shape[0]]
            if phat_int.shape[0] < envelope.shape[0]:
                phat_int = np.vstack([phat_int,
                                      np.repeat(phat_int[-1:], envelope.shape[0] - phat_int.shape[0], axis=0)])
        else:
            phat_int = np.repeat(phat[:1], envelope.shape[0], axis=0)

        mu, sigma = phat_int[:, 0], phat_int[:, 1]
        mode = np.exp(mu - sigma ** 2)
        median = np.exp(mu)
        mean = np.exp(mu + sigma ** 2 / 2)

        prah = np.zeros((envelope.shape[0], 2))
        prah[:, 0] = k1 * (mode + median) - k3 * (mean - mode)
        prah[:, 1] = (k2 * (mode + median) - k3 * (mean - mode)) if k2 != k1 else prah[:, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_env = np.log(np.where(envelope > 0, envelope, np.nan))
            cdf = 0.5 + 0.5 * erf((log_env - mu) / np.sqrt(2 * sigma ** 2))
            pdf = np.exp(-0.5 * ((log_env - mu) / sigma) ** 2) / (envelope * sigma * np.sqrt(2 * np.pi))
        cdf = np.nan_to_num(cdf)
        pdf = np.nan_to_num(pdf)

        mh = self._local_maxima(envelope, prah[:, 0], fs)
        mh = self._detection_union(mh, envelope, self.polyspike_union_time * fs)
        if k2 != k1:
            ml = self._local_maxima(envelope, prah[:, 1], fs)
            ml = self._detection_union(ml, envelope, self.polyspike_union_time * fs)
        else:
            ml = mh
        return envelope, mh, ml, prah, cdf, pdf

    # -------------------------------------------------- maxima / union helpers
    @staticmethod
    def _runs(mask):
        """Start (inclusive) / stop (exclusive) indices of True runs in a boolean mask."""
        m = mask.astype(int)
        starts = np.flatnonzero(np.diff(np.concatenate(([0], m))) > 0)
        stops = np.flatnonzero(np.diff(np.concatenate((m, [0]))) < 0) + 1
        return starts, stops

    def _local_maxima(self, envelope, prah, fs):
        above = envelope > prah
        starts, stops = self._runs(above)

        marker = np.zeros(envelope.shape[0], dtype=bool)
        for s, e in zip(starts, stops):
            if e - s > 2:
                seg = envelope[s:e]
                sgn = np.sign(np.diff(seg))
                loc = np.flatnonzero(np.diff(np.concatenate(([0], sgn))) < 0)
                marker[s + loc] = True
            else:
                marker[s + int(np.argmax(envelope[s:e]))] = True

        # union runs of maxima closer than polyspike_union_time
        pointer = np.flatnonzero(marker)
        pu = self.polyspike_union_time * fs
        state = False
        start = 0
        for k in range(len(pointer)):
            hi = int(np.ceil(pointer[k] + pu))
            seg = marker[pointer[k] + 1: hi + 1] if hi < marker.shape[0] else marker[pointer[k] + 1:]
            if state:
                if seg.sum() > 0:
                    state = True
                else:
                    state = False
                    marker[start:pointer[k]] = True
            else:
                if seg.sum() > 0:
                    state = True
                    start = pointer[k]

        # keep only the local-max peaks within each (now unioned) run
        starts, stops = self._runs(marker)
        for s, e in zip(starts, stops):
            if e - s > 1:
                lm = pointer[(pointer >= s) & (pointer < e)]
                marker[s:e] = False
                if lm.size:
                    vals = envelope[lm]
                    keep = np.flatnonzero(np.diff((np.diff(np.concatenate(([0], vals, [0]))) < 0).astype(int)) > 0)
                    marker[lm[keep]] = True
        return marker

    def _detection_union(self, marker, envelope, union_samples):
        u = int(np.ceil(union_samples))
        if u % 2 == 0:
            u += 1
        mask = np.ones(u)
        dil = np.convolve(marker.astype(float), mask, mode='same') > 0     # dilation
        ero = np.convolve((~dil).astype(float), mask, mode='same') > 0     # erosion
        closed = ~ero

        out = np.zeros(marker.shape[0], dtype=bool)
        starts, stops = self._runs(closed)
        for s, e in zip(starts, stops):
            out[s + int(np.argmax(envelope[s:e]))] = True
        return out

    # ----------------------------------------------------------- output plumbing
    def _markers_to_out(self, markers_high, markers_low, cdf, pdf, fs):
        out = _empty_out()
        t_dur = 0.005
        obvious_any = markers_high.any(axis=1)
        for ch in range(markers_high.shape[1]):
            idx = np.flatnonzero(markers_high[:, ch])
            if idx.size:
                _append_out(out, idx / fs, t_dur, ch, 1.0, cdf[idx, ch], pdf[idx, ch])
        if self.k2 != self.k1:
            for ch in range(markers_low.shape[1]):
                idx = np.flatnonzero(markers_low[:, ch] & ~markers_high[:, ch])
                for i in idx:
                    lo = max(int(i - 0.01 * fs), 0)
                    if obvious_any[lo:i + 1].any():   # ambiguous accepted near an obvious spike
                        _append_out(out, np.array([i / fs]), t_dur, ch, 0.5,
                                    np.array([cdf[i, ch]]), np.array([pdf[i, ch]]))
        order = np.argsort(out['pos']) if len(out['pos']) else []
        for key in out:
            out[key] = np.asarray(out[key])[order] if len(out[key]) else np.array([])
        return out

    def _group_discharges(self, out, envelope, background, cdf, pdf, d_decim, fs):
        nch = envelope.shape[1]
        disc = {k: [] for k in ('MV', 'MA', 'MP', 'MD', 'MW', 'MPDF')}
        if not len(out['pos']):
            return {k: np.zeros((0, nch)) for k in disc}

        tol = int(round(self.discharge_tol * fs))
        M = np.zeros((envelope.shape[0], nch))
        for p, ch, con in zip(out['pos'], out['chan'], out['con']):
            s = int(round(p * fs))
            M[s:s + tol + 1, int(ch)] = con

        active = M.sum(axis=1) > 0
        starts, stops = self._runs(active)
        for s, e in zip(starts, stops):
            seg = M[s:e]
            mv = seg.max(axis=0)
            env_seg = envelope[s:e] - background[s:e, :, 0] / self.k1
            ma = np.abs(env_seg).max(axis=0)
            mw = cdf[s:e].max(axis=0)
            mpdf = (pdf[s:e] * (seg > 0)).max(axis=0)
            mp = np.full(nch, np.nan)
            rows, cols = np.where(seg > 0)
            for rr, cc in zip(rows, cols):
                if np.isnan(mp[cc]):
                    mp[cc] = (s + rr) / fs
            disc['MV'].append(mv)
            disc['MA'].append(ma)
            disc['MW'].append(mw)
            disc['MPDF'].append(mpdf)
            disc['MP'].append(mp)
            disc['MD'].append(np.full(nch, (e - s) / fs))
        return {k: np.array(v) for k, v in disc.items()}

    # ---------------------------------------------------------------- resampling
    def _ratio(self, fs):
        """``(up, down)`` of the resampling to ``decimation`` Hz (``(1, 1)``: none)."""
        if self.decimation in (0, None) or self.decimation == fs:
            return 1, 1
        return _rational(self.decimation / fs, who='SpikeDetectorHilbert(decimation=...)')

    def _resample(self, d, fs, target):
        """Resample ``d`` (time on axis 0) from ``fs`` to ``target`` Hz (realised ratio)."""
        if target == fs:
            return d.copy()
        up, down = _rational(target / fs, who='SpikeDetectorHilbert(decimation=...)')
        return resample_poly(d, up, down, axis=0)


# Backwards-compatible alias for the name used by the original port.
spike_detector_hilbert_v24 = SpikeDetectorHilbert


def _empty_out():
    return {'pos': [], 'dur': [], 'chan': [], 'con': [], 'weight': [], 'pdf': []}


def _append_out(out, pos, dur, chan, con, weight, pdf):
    pos = np.atleast_1d(pos)
    out['pos'] = np.concatenate([out['pos'], pos])
    out['dur'] = np.concatenate([out['dur'], np.full(pos.size, dur)])
    out['chan'] = np.concatenate([out['chan'], np.full(pos.size, chan)])
    out['con'] = np.concatenate([out['con'], np.full(pos.size, con)])
    out['weight'] = np.concatenate([out['weight'], np.atleast_1d(weight)])
    out['pdf'] = np.concatenate([out['pdf'], np.atleast_1d(pdf)])
