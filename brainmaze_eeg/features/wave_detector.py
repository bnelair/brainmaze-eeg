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
those waves: rate, amplitude, peak-to-peak, duration and slope.

It is a general half-wave detector: with ``fband=(0.5, 4)`` it detects delta waves,
with ``(0.5, 0.9)`` slow oscillations, and any other band works too. The interface
follows :class:`brainmaze_eeg.features.time_domain_features.TimeDomainFeatureExtractor`
(and, for the ``(values, names)`` return,
:class:`brainmaze_eeg.features.feature_extraction.SleepSpectralFeatureExtractor`):
configure it once with ``fs`` and ``segm_size``, call it on a 1-D signal or an
``(n_channels, n_samples)`` array, and concatenate the wave features with spectral /
time-domain features for the same windows.

Two ways to use it
------------------

Windowed feature extraction (``__call__``), e.g. a 30-minute, 32-channel segment::

    from brainmaze_eeg.features.wave_detector import WaveDetector

    det = WaveDetector(fs=500, fband=(0.5, 4.0), segm_size=30, datarate=True,
                       features_on='both')
    values, names = det(x)        # x: (32, 900000) -> each value (32, 60)
    # names: DATA_RATE, WAVE_RATE, WAVE_PK2PK_MEAN, ..., WAVE_PK2PK_MEAN_BAND, ...

Raw detections, for plotting or custom analysis (``detect``)::

    det = WaveDetector(fs=500, fband=(0.5, 4.0))
    w = det.detect(x[0], return_signals=True)          # dict (1-D) or list of dicts (2-D)
    plt.plot(w['x_amp']); plt.plot(w['x_band'])
    plt.plot(w['min_pos'], w['min_val'], 'v')               # unfiltered trough values
    plt.plot(w['min_pos_band'], w['min_val_band'], 'v')     # filtered trough values

Algorithm
---------

For every signal, independently and on the **whole** signal at once (window borders
never cut a wave):

1. **Gaps.** Non-finite samples (NaN, +/-inf) in ``x`` or ``measure_on`` are gaps.
   With ``nan_policy='fill'`` (default) they are filled by
   :func:`brainmaze_utils.gaps.fill_gaps` (gaps up to 0.1 s linearly, longer gaps with
   amplitude-matched pink noise with tapered edges) so the filters can run; with
   ``nan_policy='raise'`` any gap raises ``ValueError``. The fill is never measured:
   step 6 discards every wave near a gap. Clean signals skip this step entirely.
2. **Filtering.** The mean is removed, then zero-phase filters produce two signals:

   * the *filtered* (detection) signal: band-pass to ``fband``;
   * the *unfiltered* (broadband amplitude) signal: drift removal only (high-pass
     at ``0.5 * fband[0]``), or ``measure_on`` when given (mean-subtracted, used
     as-is).

   ``filter='butter'`` (default) uses a Butterworth band-pass of order
   ``filter_order`` (default 2) in second-order sections applied forward-backward
   (``scipy.signal.sosfiltfilt``): zero phase, magnitude ``|H(f)|^2`` -- gain 1 in
   the band centre, **0.5 (-6 dB) at the band edges**, -48 dB/octave outside for
   order 2. The drift high-pass is a 4th-order Butterworth at ``0.5 * fband[0]``,
   forward-backward: gain 0.996 at ``fband[0]``, 0.004 at ``fband[0] / 4``. The measured
   response matches this design to within 0.003 for fs from 200 Hz to 25 kHz (see
   ``test_wave_detector.py``). ``filter='fft'`` uses the ideal brick-wall FFT mask of
   v1.0.0 (its drift removal is a brick-wall high-pass at ``fband[0]``). The
   brick-wall filter has a sinc impulse response that rings for many
   seconds around any transient and invents waves there (one isolated 50 uV, 1 Hz
   cycle in 30 s of silence: 14 detections in v1.0.0, 5 with ``'fft'`` here, 1 with
   ``'butter'``), and it is 10-15x slower on long signals, so it is no longer the
   default.
3. **Half-waves.** The filtered signal is split at its zero crossings into negative
   and positive half-waves. A *wave* is a negative half-wave immediately followed by a
   positive one. Both half-waves must be bounded by **real** zero crossings: the
   truncated half-waves at the start and end of the signal are never paired.
4. **Positions.** The trough (minimum of the negative half-wave) and the peak (maximum
   of the positive half-wave) are located on the filtered signal. With the default
   ``refine_lowpass=0`` these positions are used for both signals. Otherwise they are
   *refined*, for the unfiltered outputs only, to the extreme of a reference
   signal (the drift-removed signal low-passed at ``refine_lowpass * fband[1]``, or
   not low-passed if ``refine_lowpass=None``) within +/- a quarter period of
   ``fband[1]``, **clamped to the wave's own half-wave** (window truncated, not
   shifted). See *Refinement* below for why this is not the default.
5. **Duration gate.** Keep waves whose trough->peak duration on the filtered signal
   lies within half a period of the band edges, with a one-sample tolerance for
   integer sample positions::

       1 / (2 * fband[1]) - 1/fs  <=  t_peak - t_trough  <=  1 / (2 * fband[0]) + 1/fs

   This gate (more than the filter) defines the *effective* band of the detected
   waves: a wave of instantaneous frequency ``f = 1 / (2 * (t_peak - t_trough))`` is
   kept iff ``f`` is within ``fband`` up to the one-sample tolerance. A sine at exactly
   ``fband[1]`` is therefore detected (v1.0.0 lost about a third of them).
6. **Edges and gaps.** A wave's *span* runs from its interpolated down-going zero
   crossing (``zero_pos_frac``) to the end of its positive half-wave (``end_pos``). A
   wave is discarded when its span starts within ``edge_margin_s`` of the first sample
   or ends within ``edge_margin_s`` of the last sample, or overlaps a gap widened by
   ``gap_margin_s`` on each side. Both margins default to ``3 / fband[0]`` (three
   periods of the lower band edge; 6 s for the 0.5 Hz edge). Reason: the zero-phase
   filters need that long to forget the signal edge or the gap fill. Measured on 1/f
   EEG (fs 250 Hz), comparing waves in 60 s cuts with the same waves in the 30 min
   recording, the share of waves whose trough moved or whose unfiltered amplitude
   changed by more than 1 uV fell from 25 % at 2-2.5 periods from the cut to about
   1 % at 2.5-3 periods and 0.4 % at 3-4 periods (band 0.5-4 Hz). Gaps of 0.5-5 s gave
   the same picture; gaps of 20 ms or less (interpolated) barely matter. Lower the
   margins (e.g. ``gap_margin_s=0.1``) if your data has many very short dropouts and
   you accept that.
7. **Morphology.** Amplitudes, durations and slopes are read at the wave's positions
   on both signals (outputs below).

Two signals: filtered and unfiltered
------------------------------------

Every wave is measured twice, on the same set of detected waves:

* **unfiltered** (broadband; keys without suffix, features ``WAVE_*``): values of the
  drift-removed input (or of ``measure_on``) -- what the EEG really looked like,
  including faster activity riding on the wave;
* **filtered** (band-passed; keys suffixed ``_band``, features suffixed ``_BAND``):
  values of the band-passed detection signal -- the band-limited component only.
  Note the filter gain: ``|H(f)|^2`` is 0.5 at the band edges, so the filtered
  amplitude of a wave near a band edge is attenuated (a pure sine at ``fband[0]``
  or ``fband[1]`` comes out at half amplitude).

``WaveDetector(features_on=...)`` selects which set the windowed features use;
``detect()`` always returns both.

Outputs (per wave, from ``detect`` / :func:`detect_waves`)
----------------------------------------------------------

``zero_pos``
    First negative sample after the down-going zero crossing (int sample index).
``zero_pos_frac``
    The same crossing linearly interpolated between the two samples that bracket it
    (fractional sample index). Durations and the downslope use this value.
``end_pos``
    Last sample of the positive half-wave (int). ``[zero_pos_frac, end_pos]`` is the
    wave's span used for edge / gap exclusion.
``min_pos``, ``max_pos``
    Trough / peak sample index (int).
``min_val``, ``max_val``
    Signal value at the trough / peak (input units, e.g. uV).
``pk2pk``
    ``max_val - min_val`` (input units).
``delta_t``
    ``(max_pos - min_pos) / fs`` (s).
``down_dur``
    ``(min_pos - zero_pos_frac) / fs`` (s), > 0.
``upslope``
    ``pk2pk / delta_t`` (input units / s).
``downslope``
    ``-min_val / down_dur`` (input units / s). Positive for a trough below zero (the
    normal case). It can be negative if the unfiltered value at the trough is above
    zero; ``amplitude_threshold`` removes such waves.
``*_band``
    The nine keys ``min_pos`` ... ``downslope`` again, measured on the filtered signal.
``gaps``
    ``(n_gaps, 2)`` ``[start, stop)`` sample indices of the non-finite runs.
``x_band``, ``x_amp``
    Only with ``return_signals=True``: the filtered and unfiltered signals (NaN in
    gaps).

Windowed features (``__call__``)
--------------------------------

See :class:`WaveDetector`. ``DATA_RATE`` (fraction of finite samples), ``WAVE_RATE``
(waves per analysable second, Hz), and per-window means of ``pk2pk``, the selected
slope, ``delta_t``, ``min_val`` and ``max_val``.

Slope conventions
-----------------

``slope='downslope'`` (default)
    Zero-crossing -> negative-trough rate, ``-min_val / (t_trough - t_zero_cross)``.
    This is the slow-wave downslope of Carvalho et al. 2024 (``SlowWaveDetect``), who
    detect on the narrow band, measure on a broadband 0.5-35 Hz trace (pass it as
    ``measure_on=``) and keep waves with a negative peak of at least 5 uV
    (``amplitude_threshold=5``). See ``demo/eeg_wave_detection/example_one_file.py``.
``slope='upslope'``
    Trough -> peak rate, ``(max_val - min_val) / (t_peak - t_trough)`` (the
    ``WAVE_SLOPE_MEAN`` of v1.0.0 before the published downslope was restored).

For ``A sin(2 pi f t)`` both equal ``4 A f``.

Refinement and amplitude bias
-----------------------------

``refine_lowpass`` decides where the unfiltered trough / peak are read:

* ``0`` (default): at the filtered signal's trough / peak. Unbiased under noise
  (100 uV pk2pk 1 Hz sine + white noise SD 10 uV, fs 200: pk2pk 101.4), and the
  downslope has no heavy tail. Sharp, non-sinusoidal extremes are underestimated
  (a 1 Hz wave with harmonics: min -1.59 vs true -1.73, max 0.74 vs 0.89).
* ``>= 1``: refined to the extreme of the drift-removed signal low-passed at
  ``refine_lowpass * fband[1]``. Follows sharp extremes, but picks noise
  excursions: pk2pk 110.7 (``4``) / 105.3 (``2``) in the example above, and the
  trough can land just after the zero crossing, which makes ``downslope`` (an
  amplitude divided by a short duration) heavy-tailed.
* ``None``: refined on the unfiltered drift-removed signal (v1.0.0 behaviour):
  pk2pk 137.6 in the example above.

On the demo recording (6.8 h Fz-Cz, NREM epochs, ``measure_on`` 0.5-35 Hz,
``amplitude_threshold=5``) the mean downslope is SO 51.8 / delta 155.0 uV/s with
``0``, 333.0 / 299.7 with ``4`` and 136.3 / 235.5 with ``None`` (v1.0.0: 186.0 /
250.2). With refinement, 1-2 % of waves have a trough < 20 ms after the zero crossing
and slopes > 1000 uV/s, which dominate the window means. Carvalho et al. report a
delta downslope of 130.8 +/- 34.8 uV/s across subjects. ``refine_lowpass`` does not
change the filtered (``_band``) outputs.

Performance
-----------

Everything is vectorised (no Python loop over samples, waves or windows); time and
memory are O(n) per signal, dominated by the three ``sosfiltfilt`` passes. Measured
on a 12-core workstation (single process), seconds per 30-minute channel: 0.06 at
250 Hz, 0.4 at 1 kHz, 1.8 at 5 kHz; v1.0.0 needed 0.4 / 2.6 / 14.3. Use
``n_processes`` to spread channels over processes.

Changes from v1.0.0 (class version 2.0.0 -> 2.1.0)
--------------------------------------------------

Butterworth instead of brick-wall FFT filter; NaN gaps no longer zero the whole
signal; truncated edge half-waves are no longer paired; waves near the signal edges
and gaps are excluded and ``WAVE_RATE`` is normalised by analysable time; positions
come from the filtered signal by default (``refine_lowpass=0``); interpolated zero
crossing; duration gate with a one-sample tolerance; ``*_band`` outputs and
``features_on``; ``return_signals``; numpy scalar ``fs`` accepted. Expect different
numbers from v1.0.0 for the same input; see the PR that introduced 2.1.0 for a
side-by-side.

References
----------
Carvalho D.Z. et al. (2024), *Non-rapid eye movement sleep slow-wave activity features
are associated with amyloid accumulation in older adults with obstructive sleep apnoea*,
Brain Communications 6(5): fcae354. https://doi.org/10.1093/braincomms/fcae354

Lineage: this detector is the successor of the ``SlowWaveDetect`` routine used in the
study above, generalised to an arbitrary band; ``slope='downslope'`` +
``amplitude_threshold`` + ``measure_on`` reproduce that original feature's definition.
The original ``SlowWaveDetect`` source is not part of this repository, so numerical
identity with the published values cannot be verified here.
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
_WAVE_KEYS = (('zero_pos', 'zero_pos_frac', 'end_pos') + _MORPH_KEYS
              + tuple(k + '_band' for k in _MORPH_KEYS))
_INT_KEYS = ('zero_pos', 'end_pos', 'min_pos', 'max_pos', 'min_pos_band', 'max_pos_band')

_FILTERS = ('butter', 'fft')
#: drift removal for filter='butter': Butterworth high-pass of this order at
#: _DRIFT_FACTOR * fband[0], forward-backward. Gain at fband[0] = 1/(1 + 0.5**8) = 0.996,
#: so in-band amplitudes are preserved to < 0.4 %; gain at fband[0]/4 = 0.004.
_DRIFT_ORDER = 4
_DRIFT_FACTOR = 0.5
_NAN_POLICIES = ('fill', 'raise')
#: default edge / gap exclusion margin, in periods of fband[0] (see module docstring,
#: "Edges and gaps", for the measurements behind this value)
_MARGIN_PERIODS = 3.0


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
    Zero-phase SOS filtering (``sosfiltfilt``) with an odd-extension pad of 3 periods
    of ``f_low`` (the filter's lowest cutoff), capped by the signal length.
    """
    n = x.shape[0]
    padlen = min(n - 1, int(round(3.0 * fs / f_low)))
    return sosfiltfilt(sos, x, padlen=max(padlen, 0))


def _filter_signals(x0, fs, f_low, f_high, filter, order, refine_lowpass):
    """
    Return ``(x_narrow, x_hp, x_refine)``:

    * ``x_narrow`` -- detection signal, band-passed to ``(f_low, f_high)``;
    * ``x_hp``     -- drift-removed signal, the default unfiltered amplitude signal
      (``'butter'``: 4th-order Butterworth high-pass at ``0.5 * f_low``; ``'fft'``:
      brick-wall high-pass at ``f_low``);
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


def _argext_groups(v, starts, want):
    """
    Index (into ``v``) of the min (``want='min'``) or max (``want='max'``) of every
    group, where the groups are the contiguous runs ``v[starts[k]:starts[k+1]]`` (the
    last one runs to the end of ``v``). ``starts`` must be strictly increasing, start
    at 0, and every group must be non-empty. ``v`` must be finite.

    O(len(v)) via ``ufunc.reduceat``. Ties resolve to the first occurrence, matching
    ``np.argmin`` / ``np.argmax``.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)
    red = np.minimum if want == 'min' else np.maximum
    ext = red.reduceat(v, starts)
    lengths = np.diff(np.append(starts, v.size))
    hit = np.flatnonzero(v == np.repeat(ext, lengths))
    gid = np.searchsorted(starts, hit, side='right') - 1
    first = np.ones(hit.size, dtype=bool)
    first[1:] = gid[1:] != gid[:-1]
    out = hit[first]
    if out.size != starts.size:  # only possible with non-finite input
        raise RuntimeError('internal error: non-finite values in a wave segment')
    return out


def _refine_positions(x_ref, positions, half_win, want, lo_bound, hi_bound):
    """
    Move each position to the extreme of ``x_ref`` within
    ``[pos - half_win, pos + half_win]`` intersected with ``[lo_bound, hi_bound]``
    (inclusive, per position). The window is **truncated**, never shifted, at the
    bounds, so it stays centred on the original position.

    The windows must be disjoint and increasing (true here: each one lies inside its
    own half-wave). Cost and memory are O(total window length) <= O(n), independent of
    ``fs`` / ``half_win`` per wave.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0 or half_win <= 0:
        return positions
    lo = np.maximum(positions - half_win, lo_bound)
    hi = np.minimum(positions + half_win, hi_bound)
    lengths = hi - lo + 1                                    # >= 1: pos is inside [lo, hi]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    idx = np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(lo - offsets, lengths)
    return idx[_argext_groups(x_ref[idx], offsets, want)]


def _find_wave_pairs(x_narrow, x_ref, fs, f_low, f_high):
    """
    Detect (trough, peak, preceding zero-crossing) triples on the band-passed signal.

    See steps 3-5 of the module docstring. Only waves whose negative and positive
    half-waves are both bounded by real zero crossings are returned. Fully vectorised,
    O(n) time and memory.

    Returns
    -------
    trough_pos, peak_pos : np.ndarray[int]
        Refined trough / peak positions (on ``x_ref``).
    zero_pos : np.ndarray[int]
        First negative sample after the positive->negative crossing.
    zero_frac : np.ndarray[float]
        Linearly interpolated position of the positive->negative crossing.
    band_trough, band_peak : np.ndarray[int]
        Trough / peak positions on ``x_narrow`` (before refinement).
    end_pos : np.ndarray[int]
        Last sample of the positive half-wave (the sample before the next
        positive->negative crossing). ``[zero_frac, end_pos]`` is the wave's full span.
    """
    empty = np.empty(0, dtype=np.int64)
    out_empty = (empty, empty.copy(), empty.copy(), np.empty(0, dtype=np.float64),
                 empty.copy(), empty.copy(), empty.copy())
    n = x_narrow.shape[0]
    if n < 3:
        return out_empty

    s = _forward_fill_sign(x_narrow)
    change = np.flatnonzero(s[1:] != s[:-1]) + 1            # first sample of each segment
    if change.size == 0:
        return out_empty

    seg_starts = np.concatenate(([0], change))
    seg_ends = np.concatenate((change, [n])) - 1             # inclusive
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

    trough_pos = _argext_groups(x_narrow, seg_starts, 'min')[neg_idx]
    peak_pos = _argext_groups(x_narrow, seg_starts, 'max')[neg_idx + 1]
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

    end_pos = seg_ends[neg_idx + 1]
    if x_ref is x_narrow:                                    # refine_lowpass=0: no refinement
        return (band_trough.copy(), band_peak.copy(), zero_pos, zero_frac,
                band_trough, band_peak, end_pos)

    # refine on the reference within +/- a quarter period of f_high, clamped to the
    # wave's own half-wave
    half_win = int(round(fs / (4.0 * f_high)))
    trough_pos = _refine_positions(x_ref, band_trough, half_win, 'min',
                                   seg_starts[neg_idx], seg_ends[neg_idx])
    peak_pos = _refine_positions(x_ref, band_peak, half_win, 'max',
                                 seg_starts[neg_idx + 1], seg_ends[neg_idx + 1])
    return trough_pos, peak_pos, zero_pos, zero_frac, band_trough, band_peak, end_pos


def _empty_detection():
    d = {k: np.empty(0, dtype=np.float64) for k in _WAVE_KEYS}
    for k in _INT_KEYS:
        d[k] = np.empty(0, dtype=np.int64)
    return d


def _resolve_margin(margin_s, f_low, name):
    """``None`` -> ``_MARGIN_PERIODS`` periods of the lower band edge, in seconds."""
    if margin_s is None:
        return _MARGIN_PERIODS / f_low
    if isinstance(margin_s, bool) or not isinstance(margin_s, numbers.Real) \
            or not np.isfinite(margin_s) or margin_s < 0:
        raise ValueError(f'{name} must be None or a non-negative number of seconds. '
                         f'Got: {margin_s!r}')
    return float(margin_s)


def _excluded_mask(n, gaps, fs, gap_margin_s, edge_margin_s):
    """
    Boolean mask (length ``n``) of samples that are *not analysable*: inside a gap
    widened by ``gap_margin_s`` on each side, or within ``edge_margin_s`` of either end
    of the signal.

    Uses exactly the inequalities of the per-wave exclusion in :func:`detect_waves` (a
    sample at time ``t = i / fs`` is excluded iff ``lo <= t < hi`` for a widened gap
    ``[lo, hi)``, or ``t < edge_margin_s``, or ``t > (n - 1) / fs - edge_margin_s``), so
    ``WAVE_RATE`` = waves / analysable time is self-consistent.
    """
    t = np.arange(n) / fs
    bad = (t < edge_margin_s) | (t > (n - 1) / fs - edge_margin_s)
    g = np.asarray(gaps, dtype=np.float64).reshape(-1, 2)
    if g.size:
        lo = np.ceil((g[:, 0] / fs - gap_margin_s) * fs).astype(np.int64)
        hi = np.ceil((g[:, 1] / fs + gap_margin_s) * fs).astype(np.int64)
        lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
        delta = np.zeros(n + 1, dtype=np.int64)
        np.add.at(delta, lo, 1)
        np.add.at(delta, hi, -1)
        bad |= np.cumsum(delta[:-1]) > 0
    return bad


def detect_waves(x, fs, fband=(0.5, 4.0), measure_on=None, filter='butter',
                 filter_order=2, nan_policy='fill', gap_margin_s=None,
                 edge_margin_s=None, refine_lowpass=0, return_signals=False):
    """
    Detect waves in a single 1-D signal and return their positions and morphology.

    Parameters
    ----------
    x : np.ndarray
        1-D signal (any amplitude unit; outputs use the same unit, e.g. uV).
    fs : float
        Sampling frequency (Hz). Python or numpy real scalar.
    fband : (float, float)
        ``(low, high)`` band in Hz, ``0 < low < high < fs/2``.
    measure_on : np.ndarray, optional
        Signal the *broadband* amplitudes/slopes are read from (same length as ``x``).
        Detection and position refinement always run on ``x``; when ``measure_on`` is
        given, the broadband amplitudes are read from it (mean-subtracted, not filtered)
        at the refined positions, e.g. detect on a narrow band, measure on a 0.5-35 Hz
        trace. Defaults to the drift-removed ``x``. The ``*_band`` outputs always come
        from the band-passed ``x``.
    filter : {'butter', 'fft'}
        Band-pass implementation (module docstring, step 2). Default ``'butter'``.
    filter_order : int
        Butterworth order (ignored for ``'fft'``). The filter is applied
        forward-backward, so the magnitude response is squared. Default 2.
    nan_policy : {'fill', 'raise'}
        Non-finite samples in ``x`` or ``measure_on``: ``'fill'`` (default) fills them
        for filtering and discards every wave that overlaps them (plus
        ``gap_margin_s``); ``'raise'`` raises ``ValueError``.
    gap_margin_s : float or None
        Seconds added on both sides of each gap; a wave whose span overlaps the widened
        gap is discarded. ``None`` (default): ``3 / fband[0]``, three periods of the
        lower band edge (the filters' memory: values that close to a gap still depend
        on the fill; see module docstring, *Edges and gaps*).
    edge_margin_s : float or None
        A wave whose span starts within ``edge_margin_s`` of the first sample or ends
        within it of the last sample is discarded (the zero-phase filter's edge
        transient shifts zero crossings and amplitudes there). ``None`` (default):
        ``3 / fband[0]``. ``0`` keeps every complete wave.
    refine_lowpass : float or None
        Where the unfiltered trough / peak are read (module docstring, *Refinement*).
        ``0`` (default): at the filtered signal's trough / peak. ``>= 1``: refined to
        the extreme of the drift-removed ``x`` low-passed at
        ``refine_lowpass * fband[1]``. ``None``: refined on the drift-removed ``x``
        without low-pass (v1.0.0 behaviour). Does not affect the ``_band`` outputs.
    return_signals : bool
        Also return the band-passed signal (``'x_band'``) and the broadband amplitude
        signal (``'x_amp'``), both float arrays of the input length. Gaps are NaN in
        both (the fill is never returned). Default False.

    Returns
    -------
    dict
        Per-wave arrays (see *Outputs* in the module docstring): ``zero_pos,
        zero_pos_frac, end_pos`` and, for the broadband signal, ``min_pos, min_val,
        max_pos, max_val, pk2pk, delta_t, upslope, down_dur, downslope``, and the same
        nine keys with suffix ``_band`` for the band-passed signal. Plus ``gaps``:
        ``(n_gaps, 2)`` ``[start, stop)`` sample indices of the non-finite runs (union
        over ``x`` and ``measure_on``). Waves are ordered by time.
    """
    fs, f_low, f_high = _validate_band(fs, fband)
    if filter not in _FILTERS:
        raise ValueError(f"filter must be one of {_FILTERS}. Got: {filter!r}")
    if isinstance(filter_order, bool) or not isinstance(filter_order, numbers.Integral) \
            or filter_order < 1:
        raise ValueError(f'filter_order must be a positive integer. Got: {filter_order}')
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}. Got: {nan_policy!r}")
    gap_margin_s = _resolve_margin(gap_margin_s, f_low, 'gap_margin_s')
    edge_margin_s = _resolve_margin(edge_margin_s, f_low, 'edge_margin_s')

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f'detect_waves expects a 1-D signal. Got shape {x.shape}.')
    invalid = ~np.isfinite(x)

    amp_raw = None
    if measure_on is not None:
        amp_raw = np.asarray(measure_on, dtype=np.float64)
        if amp_raw.shape != x.shape:
            raise ValueError(
                f'measure_on shape {amp_raw.shape} must match x shape {x.shape}.')
        invalid = invalid | ~np.isfinite(amp_raw)

    has_gap = bool(invalid.any())
    gaps = find_gaps(np.where(invalid, np.nan, 0.0)) if has_gap \
        else np.empty((0, 2), dtype=np.int64)
    if has_gap and nan_policy == 'raise':
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
    xf = fill_gaps(np.where(invalid, np.nan, x), fs) if has_gap else x
    x0 = xf - xf[~invalid].mean()

    x_narrow, x_hp, x_refine = _filter_signals(x0, fs, f_low, f_high, filter,
                                               filter_order, refine_lowpass)

    if amp_raw is None:
        amp = x_hp
    else:
        amp = amp_raw - amp_raw[~invalid].mean()

    (trough_pos, peak_pos, zero_pos, zero_frac, b_trough, b_peak,
     end_pos) = _find_wave_pairs(x_narrow, x_refine, fs, f_low, f_high)

    # exclusion: span [zero crossing, end of the positive half-wave] must not touch a
    # widened gap or the edge zones (same inequalities as _excluded_mask)
    if trough_pos.size:
        n = x.shape[0]
        bad = (zero_frac / fs < edge_margin_s) | (end_pos / fs > (n - 1) / fs - edge_margin_s)
        if gaps.size:
            bad |= mask_in_gaps(zero_frac / fs, gaps, fs=fs, margin_s=gap_margin_s,
                                end_s=end_pos / fs)
        keep = ~bad
        trough_pos, peak_pos, b_trough, b_peak = (trough_pos[keep], peak_pos[keep],
                                                  b_trough[keep], b_peak[keep])
        zero_pos, zero_frac, end_pos = zero_pos[keep], zero_frac[keep], end_pos[keep]

    out.update({'zero_pos': zero_pos, 'zero_pos_frac': zero_frac, 'end_pos': end_pos})
    out.update(_morphology(amp, trough_pos, peak_pos, zero_frac, fs, ''))
    out.update(_morphology(x_narrow, b_trough, b_peak, zero_frac, fs, '_band'))
    if return_signals:
        out['x_band'] = np.where(invalid, np.nan, x_narrow)
        out['x_amp'] = np.where(invalid, np.nan, amp)
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
    if not (np.isfinite(f_low) and np.isfinite(f_high) and 0 < f_low < f_high):
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

    Same calling convention as
    :class:`brainmaze_eeg.features.time_domain_features.TimeDomainFeatureExtractor`:
    configure once with ``fs`` and ``segm_size``, then call it on a 1-D signal or an
    ``(n_channels, n_samples)`` array (e.g. a 30-minute multichannel segment) and get
    ``(values, names)`` back, one value per feature window. See the module docstring for
    the algorithm and the exact definition of every output.

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
        one window. Only complete windows are returned (a trailing partial window is
        dropped, like :func:`brainmaze_utils.signal.buffer` and
        ``TimeDomainFeatureExtractor``); a signal shorter than one window gives zero
        windows.
    overlap : float
        Feature window overlap in seconds. Default ``0.0``.
    slope : {'downslope', 'upslope'}
        Which slope ``WAVE_SLOPE_MEAN`` reports (see module docstring). Default
        ``'downslope'``.
    amplitude_threshold : float, optional
        Keep only waves whose broadband negative trough is at least this deep
        (``-min_val >= amplitude_threshold``, input units). ``None`` (default) keeps all
        waves. Carvalho et al. use ``5`` (uV). The same set of waves feeds the broadband
        and the band features.
    features_on : {'broadband', 'band', 'both'}
        Which signal the windowed shape features are measured on (module docstring,
        *Two signals*). ``'broadband'`` (default): ``WAVE_*`` names as in v1.0.0.
        ``'band'``: the band-passed signal, names suffixed ``_BAND``. ``'both'``: both
        sets, broadband first. ``DATA_RATE`` and ``WAVE_RATE`` appear once.
    datarate : bool
        If True, prepend a ``DATA_RATE`` feature: fraction of finite samples per window
        (finite in ``x`` and, if given, in ``measure_on``).
    n_processes : int
        Parallelise detection across signals for 2-D / list input. Default ``1``.
    filter : {'butter', 'fft'}
        Band-pass implementation. ``'butter'`` (default): zero-phase Butterworth of order
        ``filter_order``. ``'fft'``: ideal brick-wall FFT mask (the v1.0.0 filter; rings
        around transients and invents waves -- see module docstring).
    filter_order : int
        Butterworth order. Default 2.
    nan_policy : {'fill', 'raise'}
        Handling of non-finite samples. ``'fill'`` (default): fill with
        :func:`brainmaze_utils.gaps.fill_gaps`, discard waves overlapping a gap (plus
        ``gap_margin_s``), and normalise ``WAVE_RATE`` by the analysable time of each
        window. ``'raise'``: raise ``ValueError``.
    gap_margin_s, edge_margin_s : float or None
        Exclusion margins in seconds around gaps / at the two ends of each signal.
        ``None`` (default): ``3 / fband[0]``. See :func:`detect_waves`.
    refine_lowpass : float or None
        Where the unfiltered trough / peak are read. ``0`` (default): at the filtered
        signal's trough / peak; ``>= 1`` or ``None``: refined (see
        :func:`detect_waves` and the module docstring, *Refinement*).

    Notes
    -----
    Windowed features (``__call__``):

    * detection runs once on the **whole** signal (so window borders do not cut waves
      and do not add filter transients); a wave then belongs to the window containing
      its trough (``min_pos``);
    * ``WAVE_RATE`` = number of waves / *analysable* seconds in the window (Hz, i.e.
      waves per second). Analysable = finite, not within ``gap_margin_s`` of a gap and
      not within ``edge_margin_s`` of either end of the signal. ``DATA_RATE`` is the
      plain finite fraction and is unaffected by the margins. Waves whose span crosses
      into an excluded zone are discarded while their trough may lie in analysable time,
      so the rate is underestimated by at most about one wave per excluded zone border;
      exact on clean data away from the signal ends;
    * a window with no analysable sample gives ``WAVE_RATE = NaN``; a window with
      analysable data but no wave gives ``WAVE_RATE = 0``; shape features are NaN when
      there is no wave;
    * shape features (``WAVE_*_MEAN``) are means over the waves in the window, ignoring
      non-finite per-wave values (e.g. a NaN downslope).
    """

    __version__ = '2.1.0'

    _SHAPE_FEATURES = ('WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN', 'WAVE_DELTA_T_MEAN',
                       'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN')
    #: per-wave detection key behind each shape feature (slope resolved at run time)
    _SHAPE_KEYS = {'WAVE_PK2PK_MEAN': 'pk2pk', 'WAVE_SLOPE_MEAN': None,
                   'WAVE_DELTA_T_MEAN': 'delta_t', 'WAVE_MIN_MEAN': 'min_val',
                   'WAVE_MAX_MEAN': 'max_val'}
    _FEATURES_ON = ('broadband', 'band', 'both')

    def __init__(self, fs, fband=(0.5, 4.0), segm_size=None, overlap=0.0,
                 slope='downslope', amplitude_threshold=None,
                 datarate=False, n_processes=1,
                 cutoff_low=None, cutoff_high=None,
                 filter='butter', filter_order=2, nan_policy='fill',
                 gap_margin_s=None, edge_margin_s=None, refine_lowpass=0,
                 features_on='broadband'):
        # backward-compatible aliases for the old (cutoff_low, cutoff_high) signature
        if cutoff_low is not None or cutoff_high is not None:
            fband = (cutoff_low if cutoff_low is not None else fband[0],
                     cutoff_high if cutoff_high is not None else fband[1])

        fs, f_low, f_high = _validate_band(fs, fband)
        if segm_size is not None and (isinstance(segm_size, bool)
                                      or not isinstance(segm_size, numbers.Real)
                                      or not np.isfinite(segm_size) or segm_size <= 0):
            raise ValueError(f'segm_size must be a positive finite number of seconds or None. Got: {segm_size}')
        if segm_size is not None and not (0 <= overlap < segm_size):
            raise ValueError(f'overlap must be in [0, segm_size). Got: {overlap}')
        if segm_size is not None and int(round(fs * (segm_size - overlap))) < 1:
            raise ValueError('segm_size - overlap must be at least one sample.')
        if slope not in ('downslope', 'upslope'):
            raise ValueError(f"slope must be 'downslope' or 'upslope'. Got: {slope!r}")
        if amplitude_threshold is not None and (
                isinstance(amplitude_threshold, bool)
                or not isinstance(amplitude_threshold, numbers.Real)
                or not np.isfinite(amplitude_threshold)):
            raise ValueError(f'amplitude_threshold must be None or a finite number. Got: {amplitude_threshold!r}')
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
        gap_margin = _resolve_margin(gap_margin_s, f_low, 'gap_margin_s')
        edge_margin = _resolve_margin(edge_margin_s, f_low, 'edge_margin_s')
        if refine_lowpass is not None and (isinstance(refine_lowpass, bool)
                                           or not isinstance(refine_lowpass, numbers.Real)
                                           or refine_lowpass < 0):
            raise ValueError(f'refine_lowpass must be None or a number >= 0. Got: {refine_lowpass}')
        if refine_lowpass is not None and 0 < refine_lowpass < 1:
            raise ValueError('refine_lowpass must be 0 (refine on the detection signal), '
                             f'>= 1, or None. Got: {refine_lowpass}')
        if features_on not in self._FEATURES_ON:
            raise ValueError(f'features_on must be one of {self._FEATURES_ON}. Got: {features_on!r}')

        self.fs = fs
        self.fband = (f_low, f_high)
        self.segm_size = segm_size
        self.overlap = overlap
        self.slope = slope
        self.amplitude_threshold = amplitude_threshold
        self.features_on = features_on
        self.datarate = datarate
        self.n_processes = int(n_processes)
        self.filter = filter
        self.filter_order = int(filter_order)
        self.nan_policy = nan_policy
        #: resolved margins in seconds (``None`` at construction -> ``3 / fband[0]``)
        self.gap_margin_s = gap_margin
        self.edge_margin_s = edge_margin
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
                    edge_margin_s=self.edge_margin_s,
                    refine_lowpass=self.refine_lowpass)

    @property
    def feature_names(self):
        """Names returned by ``__call__``, in order."""
        names = (['DATA_RATE'] if self.datarate else []) + ['WAVE_RATE']
        if self.features_on in ('broadband', 'both'):
            names += list(self._SHAPE_FEATURES)
        if self.features_on in ('band', 'both'):
            names += [k + '_BAND' for k in self._SHAPE_FEATURES]
        return names

    # -- raw detection ------------------------------------------------------------
    def detect(self, x, measure_on=None, return_signals=False):
        """
        Raw per-signal detections.

        Parameters
        ----------
        x : np.ndarray or list
            1-D ``(n_samples,)``, 2-D ``(n_signals, n_samples)``, or list of 1-D arrays.
        measure_on : np.ndarray or list, optional
            Broadband amplitude signal(s), same shape as ``x``.
        return_signals : bool
            Also return the band-passed (``'x_band'``) and broadband amplitude
            (``'x_amp'``) signals per input signal (NaN in gaps). Default False.

        Returns
        -------
        dict or list of dict
            A single detection dict for 1-D input, otherwise one dict per signal (keys:
            see :func:`detect_waves`; broadband keys unsuffixed, band-passed keys
            suffixed ``_band``). Signals are detected **independently** -- positions
            index into that signal.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        return self._detect_list(signals, measures, single, return_signals)

    def _detect_list(self, signals, measures, single, return_signals=False):
        worker = partial(_detect_one, return_signals=return_signals, **self._detect_kwargs())
        if self.n_processes > 1 and len(signals) > 1:
            with multiprocessing.Pool(min(self.n_processes, len(signals))) as pool:
                results = pool.starmap(worker, list(zip(signals, measures)))
        else:
            results = [worker(s, m) for s, m in zip(signals, measures)]
        return results[0] if single else results

    # -- windowed feature extraction ----------------------------------------------
    def __call__(self, x, measure_on=None):
        """
        Windowed wave features, returned as ``(values, names)`` like the other extractors.

        Parameters
        ----------
        x : np.ndarray or list
            1-D ``(n_samples,)``, 2-D ``(n_channels, n_samples)``, or a list of
            equal-length 1-D arrays.
        measure_on : np.ndarray or list, optional
            Broadband amplitude signal(s), same shape as ``x``.

        Returns
        -------
        values : list of np.ndarray
            One float array per feature; shape ``(n_windows,)`` for 1-D input,
            ``(n_channels, n_windows)`` for 2-D / list input.
        names : list of str
            :attr:`feature_names`: ``[DATA_RATE?, WAVE_RATE, <shape features>]``.
            Units: ``DATA_RATE`` fraction 0-1; ``WAVE_RATE`` waves per analysable
            second (Hz); ``WAVE_PK2PK_MEAN``, ``WAVE_MIN_MEAN``, ``WAVE_MAX_MEAN``
            input units (e.g. uV); ``WAVE_SLOPE_MEAN`` input units per second;
            ``WAVE_DELTA_T_MEAN`` seconds. ``*_BAND`` features: the same, measured on
            the band-passed signal.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        detections = self._detect_list(signals, measures, False)

        names = self.feature_names
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
    def _window_bounds(self, n):
        """``(starts, ends)`` sample indices (``ends`` exclusive) of the complete windows."""
        if self.segm_size is None:
            return np.array([0]), np.array([n])
        n_segm = int(round(self.fs * self.segm_size))
        shift = int(round(self.fs * (self.segm_size - self.overlap)))
        if n < n_segm:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        starts = np.arange(0, n - n_segm + 1, shift)
        return starts, starts + n_segm

    def _features_for_signal(self, sig, measure, det):
        """All windows of one signal at once (no loop over windows or waves)."""
        n = sig.shape[0]
        starts, ends = self._window_bounds(n)

        finite = np.isfinite(sig)
        if measure is not None:
            finite &= np.isfinite(np.asarray(measure, dtype=np.float64))
        fin_cum = np.concatenate(([0], np.cumsum(finite)))
        n_finite = fin_cum[ends] - fin_cum[starts]

        ana = finite & ~_excluded_mask(n, det['gaps'], self.fs, self.gap_margin_s,
                                       self.edge_margin_s)
        ana_cum = np.concatenate(([0], np.cumsum(ana)))
        n_ana = ana_cum[ends] - ana_cum[starts]

        trough = det['min_pos']
        order = None
        if trough.size > 1 and np.any(trough[1:] < trough[:-1]):   # defensive; never expected
            order = np.argsort(trough, kind='stable')
            trough = trough[order]
        lo = np.searchsorted(trough, starts, side='left')
        hi = np.searchsorted(trough, ends, side='left')

        row = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            row['WAVE_RATE'] = np.where(n_ana > 0, (hi - lo) / (n_ana / self.fs), np.nan)
            if self.datarate:
                row['DATA_RATE'] = np.where(ends > starts, n_finite / (ends - starts), np.nan)

        suffixes = []
        if self.features_on in ('broadband', 'both'):
            suffixes.append(('', ''))
        if self.features_on in ('band', 'both'):
            suffixes.append(('_BAND', '_band'))
        for name_sfx, key_sfx in suffixes:
            for feat, key in self._SHAPE_KEYS.items():
                key = (self.slope if key is None else key) + key_sfx
                a = det[key] if order is None else det[key][order]
                row[feat + name_sfx] = _window_nanmean(a, lo, hi)
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


def _window_nanmean(a, lo, hi):
    """
    Mean of ``a[lo[k]:hi[k]]`` for every window ``k``, ignoring non-finite entries; NaN
    for a window without a finite entry. O(len(a) + n_windows) via cumulative sums.
    """
    a = np.asarray(a, dtype=np.float64)
    f = np.isfinite(a)
    cs = np.concatenate(([0.0], np.cumsum(np.where(f, a, 0.0))))
    cn = np.concatenate(([0], np.cumsum(f)))
    cnt = cn[hi] - cn[lo]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, np.nan)


def _detect_one(sig, measure, fs, fband, thr, **kwargs):
    """Detect on one signal and apply the optional negative-peak amplitude threshold."""
    det = detect_waves(sig, fs=fs, fband=fband, measure_on=measure, **kwargs)
    if thr is not None and det['min_pos'].size:
        keep = (-det['min_val']) >= thr
        for k in _WAVE_KEYS:
            det[k] = det[k][keep]
    return det
