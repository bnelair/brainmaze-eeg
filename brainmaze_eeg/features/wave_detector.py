# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
Wave detection
^^^^^^^^^^^^^^

:class:`WaveDetector` finds waves (a negative half-wave followed by a positive
half-wave) inside a chosen frequency band and reports morphological features of
those waves -- amplitude, peak-to-peak, duration and slope.

It is a general half-wave detector: with ``fband=(0.5, 4)`` it detects delta waves,
with ``(0.5, 0.9)`` slow oscillations, and any other band works too. The interface
mirrors :class:`brainmaze_eeg.features.feature_extraction.SleepSpectralFeatureExtractor`
and :class:`brainmaze_eeg.features.time_domain_features.TimeDomainFeatureExtractor`:
calling the detector returns ``(values, names)`` so wave features can be concatenated
with spectral / time-domain features for the same epochs.

Two ways to use it
------------------

Windowed feature extraction (``__call__``)::

    from brainmaze_eeg.features.wave_detector import WaveDetector

    det = WaveDetector(fs=200, fband=(0.5, 4.0), segm_size=30)
    values, names = det(x)                 # x: 1-D or (n_signals, n_samples)

Raw detections, for plotting or custom analysis (``detect``)::

    det = WaveDetector(fs=200, fband=(0.5, 4.0))
    waves = det.detect(x)                  # dict (1-D) or list of dicts (2-D)
    plt.plot(x)
    plt.plot(waves['min_pos'], waves['min_val'], 'v')
    plt.plot(waves['max_pos'], waves['max_val'], '^')

Algorithm
---------

For every signal, independently:

1. **Gaps.** Non-finite samples (NaN, +/-inf) are gaps. With ``nan_policy='fill'``
   (default) they are filled by :func:`brainmaze_utils.gaps.fill_gaps` (short gaps
   linearly, long gaps with amplitude-matched pink noise) so the filters can run; with
   ``nan_policy='raise'`` any gap raises ``ValueError``. The fill is never measured:
   step 6 discards every wave that touches a gap.
2. **Filtering.** The mean is removed, then two zero-phase filters are applied:

   * the *detection* signal: band-pass to ``fband``;
   * the *reference* signal: drift removal (high-pass at ``fband[0]``) plus a moderate
     low-pass at ``refine_lowpass * fband[1]`` (see *Refinement* below).

   ``filter='butter'`` (default) uses a Butterworth band-pass of order ``filter_order``
   applied forward-backward (``scipy.signal.sosfiltfilt``, so the magnitude response is
   squared: -6 dB at the cutoffs, zero phase). ``filter='fft'`` uses the ideal brick-wall
   FFT mask of earlier versions. The brick-wall filter has a sinc impulse response that
   rings for many seconds around any transient and invents waves there (one isolated
   50 uV, 1 Hz cycle in 30 s of silence gave 14 detections with ``'fft'`` vs 1 with
   ``'butter'``), which is why it is no longer the default.
3. **Half-waves.** The detection signal is split at its zero crossings into negative
   and positive half-waves. A *wave* is a negative half-wave immediately followed by a
   positive one. Both half-waves must be bounded by **real** zero crossings: the
   truncated half-waves at the start and end of the signal are never paired.
4. **Positions.** The trough (minimum of the negative half-wave) and the peak (maximum
   of the positive half-wave) are located on the detection signal, then *refined* to the
   extreme of the reference signal within +/- a quarter period of ``fband[1]``,
   **clamped to the wave's own half-wave** (the trough never leaves its negative
   half-wave, the peak never leaves its positive half-wave). The refinement window is
   truncated (not shifted) at the half-wave borders.
5. **Duration gate.** Keep waves whose trough->peak duration lies within half a period
   of the band edges, with a tolerance of one sample for integer sample positions::

       1 / (2 * fband[1]) - 1/fs  <=  t_peak - t_trough  <=  1 / (2 * fband[0]) + 1/fs

   This (not the filter) is what defines the *effective* band of the detected waves:
   a wave of instantaneous frequency ``f`` (= 1 / (2 * (t_peak - t_trough))) is kept iff
   ``fband[0] - tol <= f <= fband[1] + tol`` with ``tol`` from the one-sample tolerance.
6. **Gap exclusion.** A wave whose span (interpolated zero crossing -> peak) overlaps a
   gap widened by ``gap_margin_s`` on each side is discarded.
7. **Morphology.** Amplitudes are read at the refined positions on the *amplitude
   signal*: the drift-removed (high-passed, *not* low-passed) signal, or ``measure_on``
   when given (mean-subtracted, otherwise used as-is).

Outputs (per wave)
------------------

``min_pos``, ``max_pos``
    Trough / peak sample index (int).
``zero_pos``
    First sample after the preceding positive->negative zero crossing (int).
``zero_pos_frac``
    The same crossing linearly interpolated between the two samples that bracket it
    (fractional sample index, float). Durations and the downslope use this value.
``min_val``, ``max_val``
    Amplitude-signal value at the trough / peak (input units, e.g. uV).
``pk2pk``
    ``max_val - min_val``.
``delta_t``
    ``(max_pos - min_pos) / fs`` (s).
``down_dur``
    ``(min_pos - zero_pos_frac) / fs`` (s), always > 0.
``upslope``
    ``pk2pk / delta_t`` (units/s).
``downslope``
    ``-min_val / down_dur`` (units/s). Positive for a trough below zero (the normal
    case); it is negative if the amplitude signal at the trough is above zero (possible
    with ``measure_on``). ``amplitude_threshold`` removes such waves.

Slope conventions
-----------------

``slope='upslope'``
    Trough -> peak rate, ``(max_val - min_val) / (t_peak - t_trough)``. This is the
    historical behaviour of this class.
``slope='downslope'``
    Zero-crossing -> negative-trough rate, ``-min_val / (t_trough - t_zero_cross)``.
    This is the slow-wave downslope used by Carvalho et al. 2024, who measure it on a
    broadband trace (pass that trace via ``measure_on=``) after detecting on the narrow
    band, and apply an amplitude threshold on the negative peak (``amplitude_threshold``).

Refinement and amplitude bias
-----------------------------

Positions are refined on a reference that is drift-removed and low-passed at
``refine_lowpass * fband[1]`` (default 4x the upper band edge). Refining on the fully
broadband signal (``refine_lowpass=None``, the behaviour before v2.1) lets the window
argmin/argmax pick noise excursions, so ``min_val`` / ``max_val`` / ``pk2pk`` are biased
outward by noise (white noise SD 10 on a 100 uV pk2pk 1 Hz sine at 200 Hz: pk2pk 136.8);
refining on the detection signal only (``refine_lowpass=0``) is unbiased for sines but
misses sharp, non-sinusoidal troughs. The default is the compromise; see
``test_wave_detector.py`` for the measured numbers.

References
----------
Carvalho D.Z. et al. (2024), *Non-rapid eye movement sleep slow-wave activity features
are associated with amyloid accumulation in older adults with obstructive sleep apnoea*,
Brain Communications 6(5): fcae354. https://doi.org/10.1093/braincomms/fcae354

Lineage: this detector is the successor of the ``SlowWaveDetect`` routine used in the
study above, generalised to an arbitrary band; the ``slope='downslope'`` +
``amplitude_threshold`` + ``measure_on`` options reproduce that original feature.
"""

import multiprocessing
import numbers
import warnings
from functools import partial

import numpy as np
from scipy.signal import butter, sosfiltfilt

from brainmaze_utils.gaps import fill_gaps, find_gaps, mask_in_gaps

__all__ = ['WaveDetector', 'detect_waves']

#: per-wave keys of a detection dict (all arrays of length n_waves)
_MORPH_KEYS = ('min_pos', 'min_val', 'max_pos', 'max_val', 'pk2pk', 'delta_t',
               'upslope', 'down_dur', 'downslope')
_WAVE_KEYS = (('zero_pos', 'zero_pos_frac') + _MORPH_KEYS
              + tuple(k + '_band' for k in _MORPH_KEYS))
_INT_KEYS = ('zero_pos', 'min_pos', 'max_pos', 'min_pos_band', 'max_pos_band')

_FILTERS = ('butter', 'fft')
#: drift removal for filter='butter': Butterworth high-pass of this order at
#: _DRIFT_FACTOR * fband[0], forward-backward. Gain at fband[0] = 1/(1 + 0.5**8) = 0.996,
#: so in-band amplitudes are preserved to < 0.4 %; gain at fband[0]/4 = 0.004.
_DRIFT_ORDER = 4
_DRIFT_FACTOR = 0.5
_NAN_POLICIES = ('fill', 'raise')


# ----------------------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------------------
def _bandpass_fft(x, fs, f_low, f_high):
    """
    Ideal (brick-wall) FFT band-pass ``(f_low, f_high]`` in a single FFT/IFFT round trip.

    Bin frequencies use ``np.fft.fftfreq`` (true ``fs*k/n``), which also handles the
    negative-frequency mirror automatically. ``f_low=0`` gives a low-pass. Note: the
    impulse response of a brick-wall mask is a sinc, which rings for many seconds around
    every transient (see module docstring).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    Xs = np.fft.fft(x)
    freq = np.abs(np.fft.fftfreq(n, d=1.0 / fs))
    mask = (freq > f_low) & (freq <= f_high)
    return np.real(np.fft.ifft(np.where(mask, Xs, 0.0)))


def _sosfiltfilt(sos, x, fs, f_low):
    """
    Zero-phase SOS filtering with an odd-extension pad of ~3 periods of ``f_low``
    (capped by the signal length) so the edge transient of slow filters stays short.
    """
    n = x.shape[0]
    padlen = min(n - 1, int(round(3.0 * fs / f_low)))
    return sosfiltfilt(sos, x, padlen=max(padlen, 0))


def _filter_signals(x0, fs, f_low, f_high, filter, order, refine_lowpass):
    """
    Return ``(x_narrow, x_hp, x_refine)``:

    * ``x_narrow`` -- detection signal, band-passed to ``(f_low, f_high)``;
    * ``x_hp``     -- drift-removed signal (high-pass at ``f_low``), the default
      amplitude signal;
    * ``x_refine`` -- reference the positions are refined on: ``x_hp`` low-passed at
      ``refine_lowpass * f_high`` (``x_hp`` itself if ``refine_lowpass`` is None,
      ``x_narrow`` if it is 0, and ``x_hp`` if the cutoff reaches Nyquist).
    """
    nyq = fs / 2.0
    if filter == 'fft':
        x_hp = x0 - _bandpass_fft(x0, fs, 0.0, f_low)
        x_narrow = _bandpass_fft(x0, fs, f_low, f_high)
    else:
        sos_bp = butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
        x_narrow = _sosfiltfilt(sos_bp, x0, fs, f_low)
        f_d = _DRIFT_FACTOR * f_low
        sos_hp = butter(_DRIFT_ORDER, f_d, btype='highpass', fs=fs, output='sos')
        x_hp = _sosfiltfilt(sos_hp, x0, fs, f_d)

    if refine_lowpass is None:
        x_refine = x_hp
    elif refine_lowpass == 0:
        x_refine = x_narrow
    else:
        f_lp = refine_lowpass * f_high
        if f_lp >= 0.95 * nyq:
            x_refine = x_hp
        elif filter == 'fft':
            x_refine = _bandpass_fft(x0, fs, f_low, f_lp)
        else:
            sos_lp = butter(_DRIFT_ORDER, f_lp, btype='lowpass', fs=fs, output='sos')
            x_refine = _sosfiltfilt(sos_lp, x_hp, fs, f_lp)
    return x_narrow, x_hp, x_refine


# ----------------------------------------------------------------------------------
# vectorised core
# ----------------------------------------------------------------------------------
def _forward_fill_sign(x):
    """Sign of ``x`` with exact zeros carried forward from the previous non-zero sign."""
    s = np.sign(x).astype(np.int64)
    zero = s == 0
    if zero.any():
        idx = np.where(~zero, np.arange(s.size), 0)
        np.maximum.accumulate(idx, out=idx)
        s = s[idx]
        if s.size and s[0] == 0:                 # leading zeros: adopt first real sign
            nz = np.flatnonzero(s != 0)
            if nz.size:
                s[:nz[0]] = s[nz[0]]
            else:                                # all-zero signal -> constant sign, no crossings
                s[:] = 1
    return s


def _argext_per_segment(x, seg_ids, n_seg, want):
    """
    Index (into ``x``) of the min (``want='min'``) or max (``want='max'``) of every
    contiguous segment. Fully vectorised via a single lexsort; ties resolve to the
    first occurrence, matching ``np.argmin`` / ``np.argmax``.
    """
    out = np.full(n_seg, -1, dtype=np.int64)
    if x.size == 0:
        return out
    key = -x if want == 'max' else x
    order = np.lexsort((key, seg_ids))           # sort by segment, then by value
    ss = seg_ids[order]
    first = np.ones(ss.size, dtype=bool)
    first[1:] = ss[1:] != ss[:-1]                # first (== smallest key) per segment
    out[ss[first]] = order[first]
    return out


def _refine_positions(x_ref, positions, half_win, want, lo_bound=None, hi_bound=None):
    """
    Move each position to the extreme of ``x_ref`` within
    ``[pos - half_win, pos + half_win]`` intersected with ``[lo_bound, hi_bound]``
    (inclusive, per position; defaults: the signal). The window is **truncated**, never
    shifted, at the bounds, so it stays centred on the original position.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0 or half_win <= 0:
        return positions
    n = x_ref.shape[0]
    lo = positions - half_win
    hi = positions + half_win
    lo = np.maximum(lo, 0 if lo_bound is None else np.maximum(lo_bound, 0))
    hi = np.minimum(hi, n - 1 if hi_bound is None else np.minimum(hi_bound, n - 1))
    offs = np.arange(2 * half_win + 1)
    idx = lo[:, None] + offs[None, :]
    valid = idx <= hi[:, None]
    idx = np.minimum(idx, hi[:, None])
    vals = x_ref[idx].astype(np.float64, copy=True)
    vals[~valid] = np.inf if want == 'min' else -np.inf
    rel = vals.argmin(axis=1) if want == 'min' else vals.argmax(axis=1)
    return idx[np.arange(idx.shape[0]), rel]


def _find_wave_pairs(x_narrow, x_ref, fs, f_low, f_high):
    """
    Detect (trough, peak, preceding zero-crossing) triples on the band-passed signal.

    See steps 3-5 of the module docstring. Only waves whose negative and positive
    half-waves are both bounded by real zero crossings are returned.

    Returns
    -------
    trough_pos, peak_pos, zero_pos : np.ndarray[int]
    zero_frac : np.ndarray[float]
        Linearly interpolated position of the positive->negative crossing.
    """
    empty = np.empty(0, dtype=np.int64)
    out_empty = (empty, empty.copy(), empty.copy(), np.empty(0, dtype=np.float64),
                 empty.copy(), empty.copy())
    n = x_narrow.shape[0]
    if n < 3:
        return out_empty

    s = _forward_fill_sign(x_narrow)
    change = np.flatnonzero(np.diff(s) != 0) + 1           # first sample of each segment
    if change.size == 0:
        return out_empty

    seg_starts = np.concatenate(([0], change))
    seg_ends = np.concatenate((change, [n])) - 1             # inclusive
    seg_ids = np.zeros(n, dtype=np.int64)
    seg_ids[change] = 1
    np.cumsum(seg_ids, out=seg_ids)
    n_seg = seg_starts.size
    seg_polarity = s[seg_starts]                            # +/-1 per segment

    # pairs: negative segment i followed by positive segment i+1, where segment i starts
    # at a real crossing (i >= 1) and segment i+1 ends at a real crossing (i+1 <= n_seg-2).
    # The first and last segments are truncated by the signal borders and never paired.
    pair = (seg_polarity[:-1] < 0) & (seg_polarity[1:] > 0)
    pair[0] = False
    if n_seg >= 2:
        pair[n_seg - 2] = False
    neg_idx = np.flatnonzero(pair)
    if neg_idx.size == 0:
        return out_empty

    trough_of_seg = _argext_per_segment(x_narrow, seg_ids, n_seg, 'min')
    peak_of_seg = _argext_per_segment(x_narrow, seg_ids, n_seg, 'max')

    trough_pos = trough_of_seg[neg_idx]
    peak_pos = peak_of_seg[neg_idx + 1]
    zero_pos = seg_starts[neg_idx]                          # first negative sample
    a = x_narrow[zero_pos - 1]                              # >= 0
    b = x_narrow[zero_pos]                                  # < 0
    zero_frac = (zero_pos - 1) + a / (a - b)

    # duration gate on the band-limited positions (half periods of the band edges) with
    # a one-sample tolerance for the integer positions
    dur = peak_pos - trough_pos
    lo_bound = fs / (2.0 * f_high) - 1.0
    hi_bound = fs / (2.0 * f_low) + 1.0
    keep = (dur >= lo_bound) & (dur <= hi_bound)
    neg_idx, zero_pos, zero_frac = neg_idx[keep], zero_pos[keep], zero_frac[keep]
    band_trough, band_peak = trough_pos[keep], peak_pos[keep]

    # refine on the reference within +/- a quarter period of f_high, clamped to the
    # wave's own half-wave
    half_win = int(round(fs / (4.0 * f_high)))
    trough_pos = _refine_positions(x_ref, band_trough, half_win, 'min',
                                   seg_starts[neg_idx], seg_ends[neg_idx])
    peak_pos = _refine_positions(x_ref, band_peak, half_win, 'max',
                                 seg_starts[neg_idx + 1], seg_ends[neg_idx + 1])
    return trough_pos, peak_pos, zero_pos, zero_frac, band_trough, band_peak


def _empty_detection():
    d = {k: np.empty(0, dtype=np.float64) for k in _WAVE_KEYS}
    for k in _INT_KEYS:
        d[k] = np.empty(0, dtype=np.int64)
    return d


def detect_waves(x, fs, fband=(0.5, 4.0), measure_on=None, filter='butter',
                 filter_order=2, nan_policy='fill', gap_margin_s=0.1,
                 refine_lowpass=4.0, return_signals=False):
    """
    Detect waves in a single 1-D signal and return their positions and morphology.

    Parameters
    ----------
    x : np.ndarray
        1-D signal.
    fs : float
        Sampling frequency (Hz).
    fband : (float, float)
        ``(low, high)`` band in Hz, ``0 < low < high < fs/2``.
    measure_on : np.ndarray, optional
        Signal the amplitudes/slopes are read from (same length as ``x``). Detection and
        position refinement always run on ``x``; when ``measure_on`` is given, amplitudes
        are read from it (mean-subtracted, not filtered) at the refined positions, e.g.
        detect on a narrow band, measure on a 0.5-35 Hz broadband trace. Defaults to the
        drift-removed ``x``.
    filter : {'butter', 'fft'}
        Band-pass implementation (module docstring, step 2). Default ``'butter'``.
    filter_order : int
        Butterworth order (ignored for ``'fft'``). Applied forward-backward, so the
        effective attenuation is doubled. Default 2.
    nan_policy : {'fill', 'raise'}
        Non-finite samples in ``x`` or ``measure_on``: ``'fill'`` (default) fills them
        for filtering and discards every wave overlapping them; ``'raise'`` raises.
    gap_margin_s : float
        Seconds added on both sides of each gap before discarding overlapping waves.
    refine_lowpass : float or None
        Position refinement reference = drift-removed ``x`` low-passed at
        ``refine_lowpass * fband[1]``. ``None``: no low-pass (broadband, pre-2.1
        behaviour); ``0``: refine on the detection signal itself. Default 4.

    Returns
    -------
    dict
        Per-wave arrays (see *Outputs* in the module docstring): ``min_pos, min_val,
        max_pos, max_val, zero_pos, zero_pos_frac, pk2pk, delta_t, upslope, down_dur,
        downslope``; plus ``gaps`` -- ``(n_gaps, 2)`` ``[start, stop)`` sample indices of
        the non-finite runs that were filled (union over ``x`` and ``measure_on``).
    """
    fs, f_low, f_high = _validate_band(fs, fband)
    if filter not in _FILTERS:
        raise ValueError(f"filter must be one of {_FILTERS}. Got: {filter!r}")
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}. Got: {nan_policy!r}")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f'detect_waves expects a 1-D signal. Got shape {x.shape}.')
    invalid = ~np.isfinite(x)

    amp_raw = None
    if measure_on is not None:
        amp_raw = np.asarray(measure_on, dtype=np.float64).ravel()
        if amp_raw.shape[0] != x.shape[0]:
            raise ValueError(
                f'measure_on length ({amp_raw.shape[0]}) must match x length ({x.shape[0]}).')
        invalid = invalid | ~np.isfinite(amp_raw)

    gaps = find_gaps(np.where(invalid, np.nan, 0.0))
    if gaps.size and nan_policy == 'raise':
        raise ValueError(f'Input contains {int(invalid.sum())} non-finite samples in '
                         f'{gaps.shape[0]} gap(s) and nan_policy="raise".')

    out = _empty_detection()
    out['gaps'] = gaps
    if (~invalid).sum() < 3:
        if return_signals:
            out['x_band'] = np.full(x.shape, np.nan)
            out['x_amp'] = np.full(x.shape, np.nan)
        return out

    # gap fill only runs when there is a gap (clean data pays nothing)
    xf = fill_gaps(np.where(invalid, np.nan, x), fs) if gaps.size else x
    x0 = xf - xf[~invalid].mean()

    x_narrow, x_hp, x_refine = _filter_signals(x0, fs, f_low, f_high, filter,
                                               filter_order, refine_lowpass)

    if amp_raw is None:
        amp = x_hp
    else:
        amp = amp_raw - amp_raw[~invalid].mean()

    trough_pos, peak_pos, zero_pos, zero_frac, b_trough, b_peak = _find_wave_pairs(
        x_narrow, x_refine, fs, f_low, f_high)

    if gaps.size and trough_pos.size:
        bad = mask_in_gaps(zero_frac / fs, gaps, fs=fs, margin_s=gap_margin_s,
                           end_s=np.maximum(peak_pos, b_peak) / fs)
        trough_pos, peak_pos, b_trough, b_peak = (trough_pos[~bad], peak_pos[~bad],
                                                  b_trough[~bad], b_peak[~bad])
        zero_pos, zero_frac = zero_pos[~bad], zero_frac[~bad]

    out.update({'zero_pos': zero_pos, 'zero_pos_frac': zero_frac})
    out.update(_morphology(amp, trough_pos, peak_pos, zero_frac, fs, ''))
    out.update(_morphology(x_narrow, b_trough, b_peak, zero_frac, fs, '_band'))
    if return_signals:
        out['x_band'] = x_narrow
        out['x_amp'] = amp
    return out


def _morphology(sig, trough_pos, peak_pos, zero_frac, fs, suffix):
    """Per-wave amplitudes, durations and slopes read from ``sig`` (see module docstring)."""
    min_val = sig[trough_pos]
    max_val = sig[peak_pos]
    pk2pk = max_val - min_val
    delta_t = (peak_pos - trough_pos) / fs
    down_dur = (trough_pos - zero_frac) / fs
    with np.errstate(divide='ignore', invalid='ignore'):
        upslope = np.where(delta_t > 0, pk2pk / delta_t, np.nan)
        downslope = np.where(down_dur > 0, -min_val / down_dur, np.nan)
    vals = {'min_pos': trough_pos, 'min_val': min_val, 'max_pos': peak_pos,
            'max_val': max_val, 'pk2pk': pk2pk, 'delta_t': delta_t,
            'upslope': upslope, 'down_dur': down_dur, 'downslope': downslope}
    return {k + suffix: v for k, v in vals.items()}


def _validate_band(fs, fband):
    """Validate ``fs`` (any real number incl. numpy scalars) and ``fband``."""
    if isinstance(fs, bool) or not isinstance(fs, numbers.Real):
        raise ValueError(f'fs must be a positive real number. Got: {fs!r}')
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'fs must be a positive finite number. Got: {fs}')
    try:
        if len(fband) != 2:
            raise TypeError
        f_low, f_high = float(fband[0]), float(fband[1])
    except (TypeError, ValueError):
        raise ValueError(f'fband must be a (low, high) pair of numbers. Got: {fband!r}')
    if not (0 < f_low < f_high):
        raise ValueError(f'fband must be (low, high) with 0 < low < high. Got: {fband}')
    if f_high >= fs / 2:
        raise ValueError(f'fband high ({f_high}) must be below Nyquist ({fs / 2}).')
    return fs, f_low, f_high


# ----------------------------------------------------------------------------------
# detector
# ----------------------------------------------------------------------------------
class WaveDetector:
    """
    Band-limited wave detector and windowed feature extractor.

    See the module docstring for the algorithm and the exact definition of every output.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz). Python or numpy real scalar.
    fband : (float, float)
        Single ``(low, high)`` band in Hz, ``0 < low < high < fs/2``. One detector
        detects in one band; run several detectors for several bands. Default
        ``(0.5, 4.0)`` (delta).
    segm_size : float, optional
        Feature window length in seconds. ``None`` (default) treats the whole signal as
        one window.
    overlap : float
        Feature window overlap in seconds. Default ``0.0``.
    slope : {'downslope', 'upslope'}
        Which slope ``WAVE_SLOPE_MEAN`` reports (see module docstring). Default
        ``'downslope'``.
    amplitude_threshold : float, optional
        Keep only waves whose negative trough is at least this deep (``-min_val >=
        amplitude_threshold``), measured on the amplitude signal. ``None`` (default)
        keeps all waves. Carvalho et al. use ``5`` (µV).
    datarate : bool
        If True, prepend a ``DATA_RATE`` feature: fraction of finite samples per window
        (finite in ``x`` and, if given, in ``measure_on``).
    n_processes : int
        Parallelise detection across signals for 2-D / list input. Default ``1``.
    filter : {'butter', 'fft'}
        Band-pass implementation. ``'butter'`` (default): zero-phase Butterworth of order
        ``filter_order``. ``'fft'``: ideal brick-wall FFT mask (pre-2.1 default; rings
        around transients and can invent waves -- see module docstring).
    filter_order : int
        Butterworth order. Default 2.
    nan_policy : {'fill', 'raise'}
        Handling of non-finite samples. ``'fill'`` (default): fill with
        :func:`brainmaze_utils.gaps.fill_gaps`, discard waves overlapping a gap (plus
        ``gap_margin_s``), and normalise ``WAVE_RATE`` by the valid (finite) time of each
        window. ``'raise'``: raise ``ValueError``.
    gap_margin_s : float
        Exclusion margin around gaps in seconds. Default 0.1.
    refine_lowpass : float or None
        Position refinement reference low-pass, as a multiple of ``fband[1]``. Default 4.
        See :func:`detect_waves`.

    Notes
    -----
    Windowed features (``__call__``):

    * a wave belongs to the window containing its trough (``min_pos``);
    * ``WAVE_RATE`` = number of waves / valid seconds in the window, where valid seconds
      = finite samples / fs (equals ``DATA_RATE * segm_size``). Waves within
      ``gap_margin_s`` of a gap are discarded but that time is still counted as valid,
      so the rate around gaps is slightly underestimated (by roughly
      ``(2 * gap_margin_s + mean wave span) / valid seconds`` per gap);
    * a window with no valid sample gives ``WAVE_RATE = NaN``; a window with valid data
      but no wave gives ``WAVE_RATE = 0``; shape features are NaN when there is no wave;
    * shape features (``WAVE_*_MEAN``) are means over the waves in the window, ignoring
      non-finite per-wave values.
    """

    __version__ = '2.1.0'

    _SHAPE_FEATURES = ('WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN', 'WAVE_DELTA_T_MEAN',
                       'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN')

    def __init__(self, fs, fband=(0.5, 4.0), segm_size=None, overlap=0.0,
                 slope='downslope', amplitude_threshold=None,
                 datarate=False, n_processes=1,
                 cutoff_low=None, cutoff_high=None,
                 filter='butter', filter_order=2, nan_policy='fill',
                 gap_margin_s=0.1, refine_lowpass=4.0):
        # backward-compatible aliases for the old (cutoff_low, cutoff_high) signature
        if cutoff_low is not None or cutoff_high is not None:
            fband = (cutoff_low if cutoff_low is not None else fband[0],
                     cutoff_high if cutoff_high is not None else fband[1])

        fs, f_low, f_high = _validate_band(fs, fband)
        if segm_size is not None and (not isinstance(segm_size, numbers.Real)
                                      or segm_size <= 0 or not np.isfinite(segm_size)):
            raise ValueError(f'segm_size must be a positive finite number of seconds or None. Got: {segm_size}')
        if segm_size is not None and not (0 <= overlap < segm_size):
            raise ValueError(f'overlap must be in [0, segm_size). Got: {overlap}')
        if slope not in ('downslope', 'upslope'):
            raise ValueError(f"slope must be 'downslope' or 'upslope'. Got: {slope!r}")
        if isinstance(n_processes, bool) or not isinstance(n_processes, numbers.Integral) \
                or n_processes < 1:
            raise ValueError(f'n_processes must be a positive integer. Got: {n_processes}')
        if filter not in _FILTERS:
            raise ValueError(f"filter must be one of {_FILTERS}. Got: {filter!r}")
        if isinstance(filter_order, bool) or not isinstance(filter_order, numbers.Integral) \
                or filter_order < 1:
            raise ValueError(f'filter_order must be a positive integer. Got: {filter_order}')
        if nan_policy not in _NAN_POLICIES:
            raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}. Got: {nan_policy!r}")
        if not isinstance(gap_margin_s, numbers.Real) or gap_margin_s < 0:
            raise ValueError(f'gap_margin_s must be a non-negative number. Got: {gap_margin_s}')
        if refine_lowpass is not None and (not isinstance(refine_lowpass, numbers.Real)
                                           or refine_lowpass < 0):
            raise ValueError(f'refine_lowpass must be None or a number >= 0. Got: {refine_lowpass}')
        if refine_lowpass is not None and 0 < refine_lowpass < 1:
            raise ValueError('refine_lowpass must be 0 (refine on the detection signal), '
                             f'>= 1, or None. Got: {refine_lowpass}')

        self.fs = fs
        self.fband = (f_low, f_high)
        self.segm_size = segm_size
        self.overlap = overlap
        self.slope = slope
        self.amplitude_threshold = amplitude_threshold
        self.datarate = datarate
        self.n_processes = int(n_processes)
        self.filter = filter
        self.filter_order = int(filter_order)
        self.nan_policy = nan_policy
        self.gap_margin_s = float(gap_margin_s)
        self.refine_lowpass = refine_lowpass

    # -- backward-compatible read-only aliases ------------------------------------
    @property
    def cutoff_low(self):
        return self.fband[0]

    @property
    def cutoff_high(self):
        return self.fband[1]

    def _detect_kwargs(self):
        return dict(fs=self.fs, fband=self.fband, thr=self.amplitude_threshold,
                    filter=self.filter, filter_order=self.filter_order,
                    nan_policy=self.nan_policy, gap_margin_s=self.gap_margin_s,
                    refine_lowpass=self.refine_lowpass)

    # -- raw detection ------------------------------------------------------------
    def detect(self, x, measure_on=None):
        """
        Raw per-signal detections.

        Parameters
        ----------
        x : np.ndarray or list
            1-D ``(n_samples,)``, 2-D ``(n_signals, n_samples)``, or list of 1-D arrays.
        measure_on : np.ndarray or list, optional
            Amplitude signal(s), same shape as ``x``.

        Returns
        -------
        dict or list of dict
            A single detection dict for 1-D input, otherwise one dict per signal (keys:
            see :func:`detect_waves`). Signals are detected **independently** --
            positions index into that signal.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        return self._detect_list(signals, measures, single)

    def _detect_list(self, signals, measures, single):
        worker = partial(_detect_one, **self._detect_kwargs())
        if self.n_processes > 1 and len(signals) > 1:
            with multiprocessing.Pool(self.n_processes) as pool:
                results = pool.starmap(worker, list(zip(signals, measures)))
        else:
            results = [worker(s, m) for s, m in zip(signals, measures)]
        return results[0] if single else results

    # -- windowed feature extraction ----------------------------------------------
    def __call__(self, x, measure_on=None):
        """
        Windowed wave features, returned as ``(values, names)`` like the other extractors.

        Returns
        -------
        values : list of np.ndarray
            One array per feature; shape ``(n_windows,)`` for 1-D input,
            ``(n_signals, n_windows)`` for 2-D / list input.
        names : list of str
            ``[DATA_RATE?, WAVE_RATE, WAVE_PK2PK_MEAN, WAVE_SLOPE_MEAN, WAVE_DELTA_T_MEAN,
            WAVE_MIN_MEAN, WAVE_MAX_MEAN]``. ``WAVE_RATE`` is in waves per valid second
            (see class Notes); ``WAVE_SLOPE_MEAN`` in units/s; ``WAVE_DELTA_T_MEAN`` in s;
            amplitudes in input units.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        detections = self._detect_list(signals, measures, False)

        names = (['DATA_RATE'] if self.datarate else []) + ['WAVE_RATE'] + list(self._SHAPE_FEATURES)
        per_signal = [self._features_for_signal(sig, m, det)
                      for sig, m, det in zip(signals, measures, detections)]

        # np.stack (not np.array) so a shape mismatch between signals -- e.g. a list of
        # unequal-length signals producing different window counts -- raises a clear error
        # instead of silently building a dtype=object array.
        values = [np.stack([row[k] for row in per_signal]) for k in names]
        if single:
            values = [v[0] for v in values]
        return values, names

    # -- helpers ------------------------------------------------------------------
    def _window_starts(self, n):
        if self.segm_size is None:
            return np.array([0]), n
        n_segm = int(round(self.fs * self.segm_size))
        shift = int(round(self.fs * (self.segm_size - self.overlap)))
        if n < n_segm:
            return np.empty(0, dtype=int), n_segm
        return np.arange(0, n - n_segm + 1, shift), n_segm

    def _features_for_signal(self, sig, measure, det):
        n = sig.shape[0]
        starts, n_segm = self._window_starts(n)

        valid = np.isfinite(sig)
        if measure is not None:
            valid &= np.isfinite(np.asarray(measure, dtype=np.float64).ravel())
        valid_cum = np.concatenate(([0], np.cumsum(valid)))

        trough = det['min_pos']
        slope = det['downslope'] if self.slope == 'downslope' else det['upslope']
        row = {}
        rate, pk2pk, slp, dt, mn, mx, drate = [], [], [], [], [], [], []
        for s in starts:
            e = min(s + n_segm, n)
            n_valid = int(valid_cum[e] - valid_cum[s])
            in_win = (trough >= s) & (trough < s + n_segm)
            k = int(in_win.sum())
            rate.append(k / (n_valid / self.fs) if n_valid else np.nan)
            if k:
                pk2pk.append(_nanmean(det['pk2pk'][in_win]))
                slp.append(_nanmean(slope[in_win]))
                dt.append(_nanmean(det['delta_t'][in_win]))
                mn.append(_nanmean(det['min_val'][in_win]))
                mx.append(_nanmean(det['max_val'][in_win]))
            else:
                pk2pk.append(np.nan); slp.append(np.nan); dt.append(np.nan)
                mn.append(np.nan); mx.append(np.nan)
            if self.datarate:
                drate.append(n_valid / (e - s) if e > s else np.nan)

        row['WAVE_RATE'] = np.asarray(rate, dtype=np.float64)
        row['WAVE_PK2PK_MEAN'] = np.asarray(pk2pk, dtype=np.float64)
        row['WAVE_SLOPE_MEAN'] = np.asarray(slp, dtype=np.float64)
        row['WAVE_DELTA_T_MEAN'] = np.asarray(dt, dtype=np.float64)
        row['WAVE_MIN_MEAN'] = np.asarray(mn, dtype=np.float64)
        row['WAVE_MAX_MEAN'] = np.asarray(mx, dtype=np.float64)
        if self.datarate:
            row['DATA_RATE'] = np.asarray(drate, dtype=np.float64)
        return row

    def _as_signal_list(self, x, measure_on):
        single = False
        if isinstance(x, np.ndarray) and x.ndim == 1:
            signals = [np.asarray(x, dtype=np.float64)]
            single = True
        elif isinstance(x, np.ndarray):
            if x.ndim != 2:
                raise ValueError(f"Input 'x' must be 1-D or 2-D. Got {x.ndim}-D.")
            if x.shape[0] > x.shape[1]:
                warnings.warn(
                    f"WaveDetector: 2-D input has shape {x.shape} -- more signals than "
                    "samples. Input must be (n_signals, n_samples); did you pass "
                    "(n_samples, n_signals)? Transpose it if so.", UserWarning, stacklevel=3)
            signals = [np.asarray(row, dtype=np.float64) for row in x]
        elif isinstance(x, (list, tuple)):
            signals = [np.asarray(s, dtype=np.float64).ravel() for s in x]
        else:
            raise ValueError("Input 'x' must be a numpy array or a list of 1-D arrays.")

        if measure_on is None:
            measures = [None] * len(signals)
        elif isinstance(measure_on, np.ndarray) and measure_on.ndim == 1:
            measures = [measure_on]
        elif isinstance(measure_on, np.ndarray):
            measures = [row for row in measure_on]
        else:
            measures = list(measure_on)
        if len(measures) != len(signals):
            raise ValueError('measure_on must have the same number of signals as x.')
        return signals, measures, single


def _nanmean(a):
    """Mean over finite entries; NaN (no warning) when none are finite."""
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    return a[finite].mean() if finite.any() else np.nan


def _detect_one(sig, measure, fs, fband, thr, **kwargs):
    """Detect on one signal and apply the optional negative-peak amplitude threshold."""
    det = detect_waves(sig, fs=fs, fband=fband, measure_on=measure, **kwargs)
    if thr is not None and det['min_pos'].size:
        keep = (-det['min_val']) >= thr
        for k in _WAVE_KEYS:
            det[k] = det[k][keep]
    return det
