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
    frequency parameter made configurable. Multichannel layout ``(n_channels, n_samples)``.

:class:`SpikeDetectorHilbert` (MATLAB-v24 compatible)
    A port of the published MATLAB ``spike_detector_hilbert_v24.m`` with its full output
    (per-detection CDF/PDF weights, multichannel discharge grouping, ambiguous ``k2``
    class, segment buffering). Multichannel layout ``(n_samples, n_channels)`` (MATLAB's
    ``[samples, channels]``).

Algorithm of :func:`detect_spikes_janca`
----------------------------------------
For each channel (all filters zero-phase, ``sosfiltfilt``, at the **input** rate ``fs``):

1. Band-pass Butterworth, order ``filter_order`` (3), edges ``band`` (10, 60) Hz.
2. Band-stop Butterworth, order ``notch_order`` (3), ``powerline +/- notch_width/2``
   (50 +/- 2.5 Hz), optionally also at harmonics (``notch_harmonics``).
3. Resample with :func:`scipy.signal.resample_poly` (its anti-alias FIR runs on the
   already band-limited signal). ``decimation='integer'`` (default, reference): if
   ``target_fs`` is set and ``fs >= 2 * target_fs``, decimate by the **integer** factor
   ``q = floor(fs / target_fs)``; the analysis rate ``fs_a = fs / q`` is then generally
   *not* ``target_fs`` (500 Hz -> 250 Hz; 512 -> 256; 2048 -> 204.8; 256 or 399 Hz -> not
   decimated). ``decimation='exact'`` (ours): resample to exactly ``target_fs`` whenever
   ``fs > target_fs``.
4. Envelope ``e = |hilbert(x)|``.
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
   ``decimation='exact'``: ``round(index_a * fs / fs_a)``).

Known differences from the eeg_forge reference (all deliberate fixes)
--------------------------------------------------------------------
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
- **Validation.** Band edges, notch frequencies and the analysis Nyquist are validated
  (``ValueError``); a power-line notch that does not fit below Nyquist is skipped with a
  ``UserWarning`` (``powerline=None`` disables it silently).
- **Multichannel input** ``(n_channels, n_samples)`` (the reference is 1-D only).
- **NaN gaps** are filled before and detections in/near gaps removed after (``nan_policy``);
  the reference returns zero detections for a channel with a single NaN.
- ``min_distance_s * fs_a < 1`` is clamped to 1 sample (the reference would raise).
- **Optional exact resampling** (``decimation='exact'``); the default keeps the reference's
  integer decimation so that results are identical to it.
"""

import warnings
from fractions import Fraction

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, firwin, hilbert, resample_poly, sosfiltfilt, filtfilt
from scipy.special import erf

from brainmaze_eeg.spikes import _filters as flt
from brainmaze_eeg.spikes._gaps import in_gap_mask, prepare_signal

__all__ = ['detect_spikes_janca', 'design_janca_filters', 'janca_decimation_factor',
           'janca_resampling',
           'SpikeDetectorHilbert', 'spike_detector_hilbert_v24']


# ============================================================================ primary detector
def janca_decimation_factor(fs, target_fs=200.0):
    """
    Integer decimation factor of the reference rule (``decimation='integer'``).

    ``floor(fs / target_fs)`` when ``target_fs`` is set and ``fs >= 2 * target_fs``,
    otherwise 1 (no decimation). With the default ``target_fs=200`` this is the eeg_forge rule
    (decimate only when ``fs >= 400``). The resulting analysis rate ``fs / q`` is generally
    **not** ``target_fs`` (512 Hz -> 256 Hz, 2048 Hz -> 204.8 Hz, 399 Hz -> 399 Hz).
    """
    if target_fs is None:
        return 1
    if not target_fs > 0:
        raise ValueError(f'target_fs must be > 0 Hz or None, got {target_fs}')
    if fs >= 2 * target_fs:
        return int(np.floor(fs / target_fs))
    return 1


def _rational(ratio, max_den=10000):
    """``(up, down)`` integers with ``up / down == ratio`` (to 1e-9 relative)."""
    fr = Fraction(ratio).limit_denominator(max_den)
    if abs(fr.numerator / fr.denominator - ratio) > 1e-9 * ratio:
        raise ValueError(f'cannot express the resampling ratio {ratio!r} as a rational number '
                         f'with denominator <= {max_den}; use decimation=\'integer\'')
    return fr.numerator, fr.denominator


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
        ``'exact'``: rational resampling to exactly ``target_fs`` whenever
        ``fs > target_fs`` (never upsamples).

    Returns
    -------
    (up, down, fs_analysis) : (int, int, float)
        :func:`scipy.signal.resample_poly` factors and the analysis rate ``fs * up / down``.
    """
    if decimation not in ('integer', 'exact'):
        raise ValueError(f"decimation must be 'integer' or 'exact', got {decimation!r}")
    if decimation == 'integer' or target_fs is None:
        q = janca_decimation_factor(fs, target_fs)
        return 1, q, fs / q
    if not target_fs > 0:
        raise ValueError(f'target_fs must be > 0 Hz or None, got {target_fs}')
    if fs <= target_fs:
        return 1, 1, float(fs)
    up, down = _rational(target_fs / fs)
    return up, down, fs * up / down


def design_janca_filters(fs, band=(10.0, 60.0), filter_order=3, powerline=50.0,
                         notch_width=5.0, notch_order=3, notch_harmonics=1):
    """
    Design the filters :func:`detect_spikes_janca` applies at the input rate ``fs``.

    Parameters are those of :func:`detect_spikes_janca`.

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
    fs = float(fs)
    if not fs > 0:
        raise ValueError(f'fs must be > 0, got {fs}')
    out = {'bandpass': flt.butter_bandpass(band, fs, filter_order), 'notches': [],
           'skipped_notches': []}
    if powerline is None:
        return out
    if not powerline > 0:
        raise ValueError(f'powerline must be > 0 Hz or None, got {powerline}')
    if not 0 < notch_width < 2 * powerline:
        raise ValueError(f'notch_width must be in (0, 2*powerline), got {notch_width}')
    if int(notch_harmonics) != notch_harmonics or notch_harmonics < 1:
        raise ValueError(f'notch_harmonics must be a positive integer, got {notch_harmonics}')
    for k in range(1, int(notch_harmonics) + 1):
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


def _as_channels_first(x, name='x'):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[np.newaxis, :], True
    if x.ndim != 2:
        raise ValueError(f"'{name}' must be 1-D (n_samples,) or 2-D (n_channels, n_samples), "
                         f'got {x.ndim}-D')
    if x.shape[0] > x.shape[1]:
        raise ValueError(f"'{name}' has shape {x.shape}: more channels than samples. The "
                         'layout is (n_channels, n_samples); transpose your array (x.T).')
    return x, False


def detect_spikes_janca(x, fs, *, band=(10.0, 60.0), filter_order=3, powerline=50.0,
                        notch_width=5.0, notch_order=3, notch_harmonics=1, target_fs=200.0,
                        decimation='integer', window_s=5.0, threshold=3.65, min_distance_s=0.1, eps_rel=1e-6,
                        nan_policy='fill', gap_margin_s=0.1, fill_kwargs=None,
                        return_details=False):
    """
    Janca envelope-distribution spike detector (eeg_forge formulation, fixed).

    See the module docstring for the algorithm and the differences from the reference.

    Parameters
    ----------
    x : np.ndarray
        Signal, ``(n_samples,)`` or ``(n_channels, n_samples)``. Any amplitude unit
        (the detector is scale-invariant). NaN marks missing data (see ``nan_policy``).
    fs : float
        Sampling frequency of ``x`` (Hz).
    band : (float, float)
        Band-pass edges in Hz (reference: 10, 60). Must satisfy ``0 < low < high`` and
        ``high`` < Nyquist of both ``fs`` and the analysis rate ``fs / q``.
    filter_order : int
        Butterworth prototype order of the band-pass (reference: 3). Zero-phase: the
        effective response is -6 dB at both edges.
    powerline : float or None
        Power-line frequency in Hz (reference: 50). **Set 60 for North-American data.**
        ``None`` disables the notch.
    notch_width : float
        Total width (Hz) of the band-stop, centred on ``powerline`` (reference: 5, i.e.
        +/-2.5 Hz; -6 dB at the edges after zero-phase filtering).
    notch_order : int
        Butterworth prototype order of the band-stop (reference: 3).
    notch_harmonics : int
        Number of notches at ``k * powerline``, ``k = 1..notch_harmonics`` (reference: 1).
        Notches not fitting below Nyquist are skipped with a warning.
    target_fs : float or None
        Decimation target (Hz) (reference: 200). ``None`` keeps the input rate.
    decimation : {'integer', 'exact'}
        ``'integer'`` (default, reference): decimate by the integer factor
        ``q = floor(fs / target_fs)`` only when ``fs >= 2 * target_fs``; the analysis rate
        ``fs / q`` is then generally not ``target_fs`` (512 -> 256 Hz, 2048 -> 204.8 Hz,
        250..399 Hz -> not decimated). ``'exact'`` (ours): rational polyphase resampling
        to exactly ``target_fs`` whenever ``fs > target_fs``, so the analysis rate (and
        with it the effective time resolution of the envelope model) is the same for every
        input rate. Detections are mapped back to input samples by rounding.
    window_s : float
        Length (s) of the sliding window of the log-envelope statistics (reference ``w``: 5).
    threshold : float
        Threshold multiplier on ``mode + median`` of the local log-normal model
        (reference ``thr``: 3.65, the paper's ``k1``).
    min_distance_s : float
        Minimum distance between detections in seconds (reference: 0.1). Of two close
        maxima the larger is kept.
    eps_rel : float
        Envelope offset before the log, relative to the channel's median envelope
        (our choice, replaces the reference's absolute 1e-6; see module docstring).
    nan_policy : {'fill', 'raise'}
        ``'fill'`` (default): NaN gaps are filled with
        :func:`brainmaze_eeg.spikes._gaps.fill_gaps` before detection and detections inside a gap
        or within ``gap_margin_s`` of it are dropped. ``'raise'``: ``ValueError`` on NaN.
    gap_margin_s : float
        Exclusion margin around each gap in seconds (default 0.1).
    fill_kwargs : dict, optional
        Passed to :func:`brainmaze_eeg.spikes._gaps.fill_gaps` (e.g. ``{'max_interp_s': 0.2}``; see the README, "Gaps").
    return_details : bool
        Also return per-channel diagnostics (see Returns).

    Returns
    -------
    detections : np.ndarray or list of np.ndarray
        For 1-D input: ``int64`` array of detection **sample indices into x** (input rate),
        sorted. For 2-D input: a list with one such array per channel (row of ``x``).
        Convert to seconds with ``detections / fs``.
    details : dict or list of dict
        Only with ``return_details=True`` (one dict per channel for 2-D input):
        ``fs_analysis`` (Hz), ``up``/``down`` resampling factors (``fs_analysis = fs * up /
        down``), ``envelope`` and ``threshold`` (at the analysis rate; sample ``i``
        corresponds to input sample ``i * down / up``), ``filters``
        (output of :func:`design_janca_filters`), ``gaps`` (``[start, stop)`` input samples).

    Raises
    ------
    ValueError
        Invalid parameters, a band edge at/above the analysis Nyquist, a 2-D array with more
        rows than columns (probably transposed), ``+/-inf`` in ``x``, or NaN with
        ``nan_policy='raise'``.
    """
    fs = float(fs)
    if not fs > 0:
        raise ValueError(f'fs must be > 0, got {fs}')
    X, one_d = _as_channels_first(x)
    if not window_s > 0:
        raise ValueError(f'window_s must be > 0, got {window_s}')
    if not threshold > 0:
        raise ValueError(f'threshold must be > 0, got {threshold}')
    if not min_distance_s >= 0:
        raise ValueError(f'min_distance_s must be >= 0, got {min_distance_s}')
    if not eps_rel >= 0:
        raise ValueError(f'eps_rel must be >= 0, got {eps_rel}')
    if not gap_margin_s >= 0:
        raise ValueError(f'gap_margin_s must be >= 0, got {gap_margin_s}')

    filters = design_janca_filters(fs, band, filter_order, powerline, notch_width,
                                   notch_order, notch_harmonics)
    up, down, fs_a = janca_resampling(fs, target_fs, decimation)
    if band[1] >= fs_a / 2:
        raise ValueError(f'band high edge ({band[1]} Hz) must be < Nyquist of the analysis '
                         f'rate ({fs_a / 2} Hz; fs={fs} Hz resampled by {up}/{down}). Raise '
                         'target_fs or set target_fs=None.')
    win = int(window_s * fs_a)
    if win % 2 == 0:
        win += 1
    dist = max(int(min_distance_s * fs_a), 1)

    Y, gaps, dead = prepare_signal(X, fs, nan_policy, fill_kwargs)

    # -- preprocessing (all channels at once, time on the last axis) ----------------------
    y = sosfiltfilt(filters['bandpass'], Y, axis=-1)
    for _, sos in filters['notches']:
        y = sosfiltfilt(sos, y, axis=-1)
    if (up, down) != (1, 1):
        y = resample_poly(y, up, down, axis=-1)
    env = np.abs(hilbert(y, axis=-1))

    results, details = [], []
    for c in range(X.shape[0]):
        e = env[c]
        idx = np.zeros(0, dtype=np.int64)
        thr_curve = np.full(e.shape, np.nan)
        scale = np.median(e)
        if not scale > 0:
            scale = e.mean()
        if not dead[c] and scale > 0:
            log_e = np.log(e + eps_rel * scale)
            mu = uniform_filter1d(log_e, win, mode='reflect')
            sd = np.sqrt(uniform_filter1d((log_e - mu) ** 2, win, mode='reflect'))
            thr_curve = threshold * (np.exp(mu - sd ** 2) + np.exp(mu))
            pk = find_peaks(e, height=thr_curve, distance=dist)[0]
            if up == 1:
                idx = pk.astype(np.int64) * down
            else:
                idx = np.minimum(np.round(pk * (down / up)).astype(np.int64), X.shape[1] - 1)
            if len(gaps[c]):
                idx = idx[~in_gap_mask(idx / fs, gaps[c], fs, gap_margin_s)]
        results.append(idx)
        if return_details:
            details.append({'fs_analysis': fs_a, 'up': up, 'down': down, 'envelope': e,
                            'threshold': thr_curve, 'filters': filters, 'gaps': gaps[c]})

    if one_d:
        results = results[0]
        details = details[0] if details else details
    return (results, details) if return_details else results


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

      As in v24, Chebyshev is replaced by Butterworth (with a warning) when ``decimation``
      is set to a rate other than 200 Hz.

    Parameters (defaults follow v24)
    --------------------------------
    bandwidth : (float, float)
        Band-pass edges [low, high] in Hz. Default [10, 60].
    k1 : float
        Threshold multiplier for obvious spikes. Default 3.65.
    k2 : float
        Threshold multiplier for ambiguous spikes (accepted only near an obvious detection).
        Default equals ``k1`` (ambiguous detection disabled).
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
        Envelope-model window and overlap in **seconds**. Defaults 5 and 4.
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
    nan_policy : {'fill', 'raise'}
        NaN handling, as in :func:`detect_spikes_janca`. With ``'fill'`` the returned
        signals (``d_decim``, ``envelope``, ...) are those of the filled signal, and
        detections in/near gaps are removed before discharge grouping.
    gap_margin_s : float
        Exclusion margin around gaps (seconds). Default 0.1.

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
        self.nan_policy = 'fill'
        self.gap_margin_s = 0.1
        self.fill_kwargs = None
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f'unknown parameter {key!r}')
            setattr(self, key, value)
        if 'k2' not in kwargs:
            self.k2 = self.k1
        if self.k2 < self.k1:
            raise ValueError('k2 must be >= k1')
        if self.f_type not in (1, 2, 3):
            raise ValueError(f'f_type must be 1 (Chebyshev-II), 2 (Butterworth) or 3 (FIR), '
                             f'got {self.f_type!r}')

    # ------------------------------------------------------------------ filter design
    def _effective_f_type(self):
        if self.f_type == 1 and self.decimation not in (0, None) and self.decimation != 200:
            warnings.warn('f_type switched to Butterworth for non-200 Hz decimation (as v24)')
            return 2
        return self.f_type

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
        ftype = self._effective_f_type()
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
        fs = float(fs)
        if np.isfinite(self.beta):
            raise NotImplementedError('beta/mu rejection is not implemented')

        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            d = d[:, None]
        elif d.ndim != 2:
            raise ValueError("'d' must be 1-D or 2-D (n_samples, n_channels)")
        if d.shape[1] > d.shape[0]:
            raise ValueError(f"'d' has shape {d.shape}: more channels than samples. "
                             'SpikeDetectorHilbert expects (n_samples, n_channels); transpose '
                             'your array (d.T).')

        target = fs if self.decimation in (0, None) else float(self.decimation)
        if self.bandwidth[1] >= target / 2:
            raise ValueError(f'bandwidth high edge {self.bandwidth[1]} >= target Nyquist {target/2}')
        filters = self.design_filters(target)

        filled, gaps, _ = prepare_signal(d.T, fs, self.nan_policy, self.fill_kwargs)
        d = filled.T

        # -- decimate (per channel), then notch mains + 1 Hz high-pass -----------
        d_dec = self._resample(d, fs, target)
        fsd = target
        for _, sos in filters['notches']:
            d_dec = sosfiltfilt(sos, d_dec, axis=0)
        d_decim = sosfiltfilt(filters['highpass'], d_dec, axis=0)

        n = d_decim.shape[0]
        winsize = int(round(self.winsize * fsd))
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
        out = self._drop_in_gaps(out, gaps, fs)
        discharges = self._group_discharges(out, envelope, background, envelope_cdf,
                                            envelope_pdf, d_decim, fsd)
        return out, discharges, d_decim, envelope, background, envelope_pdf

    def _drop_in_gaps(self, out, gaps, fs):
        if not len(out['pos']) or not any(len(g) for g in gaps):
            return out
        keep = np.ones(len(out['pos']), dtype=bool)
        for ch, g in enumerate(gaps):
            if len(g):
                sel = out['chan'] == ch
                keep[sel] = ~in_gap_mask(out['pos'][sel], g, fs, self.gap_margin_s)
        return {k: v[keep] for k, v in out.items()}

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
    def _resample(self, d, fs, target):
        if target == fs:
            return d.copy()
        up, down = _rational(target / fs)
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
