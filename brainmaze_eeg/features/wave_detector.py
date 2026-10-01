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
    # names: DATA_RATE, WAVE_RATE, WAVE_PK2PK_MEAN, ..., WAVE_SLOPE_MEDIAN,
    #        WAVE_PK2PK_MEAN_BAND, ..., ANALYSABLE_RATE

Raw detections, for plotting or custom analysis (``detect``)::

    det = WaveDetector(fs=500, fband=(0.5, 4.0))
    w = det.detect(x[0], return_signals=True)          # dict (1-D) or list of dicts (2-D)
    plt.plot(w['x_amp']); plt.plot(w['x_band'])
    plt.plot(w['min_pos'], w['min_val'], 'v')               # unfiltered trough values
    plt.plot(w['min_pos_band'], w['min_val_band'], 'v')     # filtered trough values

Consecutive segments (e.g. a night cut into 30-min blocks) can pass a few seconds of the
neighbouring blocks as ``context=(n_before, n_after)``: it is used for filtering only, so
no wave is lost to the edge margin at the block borders and none is counted twice.

Algorithm
---------

For every signal, independently and on the **whole** signal at once (window borders
never cut a wave):

1. **Gaps.** Non-finite samples (NaN, +/-inf) in ``x`` or ``measure_on`` are gaps.
   With ``nan_policy='fill'`` (default) they are filled by
   :func:`brainmaze_utils.gaps.fill_gaps` with ``method='spectral'`` and
   ``max_interp_s=0.1`` (both passed explicitly): gaps up to 0.1 s are interpolated
   linearly, longer gaps get noise whose spectrum matches the neighbouring data, with
   edges conditioned on the neighbouring samples. This only lets the filters run; the fill is never measured: step 6
   discards every wave near a gap. With ``nan_policy='raise'`` any gap raises
   ``ValueError``. Clean signals skip this step entirely.
2. **Filtering.** The mean is removed, then zero-phase filters produce two signals:

   * the *filtered* (detection) signal: band-pass to ``fband``;
   * the *unfiltered* (broadband amplitude) signal: drift removal only (high-pass
     at ``0.5 * fband[0]``), or ``measure_on`` when given (mean-subtracted, used
     as-is), or, with ``trough='paper'``, the paper's trace (see *Trough placement*).

   ``filter='butter'`` (default) uses a Butterworth band-pass of order
   ``filter_order`` (default 2) in second-order sections applied forward-backward
   (``scipy.signal.sosfiltfilt``): zero phase, magnitude ``|H(f)|^2`` -- gain 1 in
   the band centre, **0.5 (-6 dB) at the band edges**, and a skirt of about
   -24 dB/octave outside the band for order 2 (12 dB/octave per pass, two passes). The
   drift high-pass is a 4th-order Butterworth at ``0.5 * fband[0]``, forward-backward:
   gain 0.996 at ``fband[0]``, 0.004 at ``fband[0] / 4``. The measured response
   matches this design to within 0.003 for fs from 200 Hz to 25 kHz (see
   ``test_wave_detector.py``). ``filter='fft'`` uses the ideal brick-wall FFT mask of
   v1.0.0 (its drift removal is a brick-wall high-pass at ``fband[0]``). The
   brick-wall filter has a sinc impulse response that rings for many seconds around
   any transient and invents waves there (one isolated 50 uV, 1 Hz cycle in 30 s of
   silence: 14 detections in v1.0.0, 5 with ``'fft'`` here, 1 with ``'butter'``), and
   it is 10-15x slower on long signals, so it is no longer the default.
3. **Half-waves.** The filtered signal (with ``trough='paper'``: the paper's trace) is
   split at its zero crossings into negative and positive half-waves. A *wave* is a
   negative half-wave immediately followed by a positive one. Both half-waves must be
   bounded by **real** zero crossings: the truncated half-waves at the start and end
   of the signal are never paired.
4. **Positions.** The trough / peak are placed according to ``trough`` (see *Trough
   placement*).
5. **Duration gate.** Keep waves whose duration lies within half a period of the band
   edges, up to a relative tolerance of 1 %::

       0.99 / (2 * fband[1])  <=  duration  <=  1.01 / (2 * fband[0])

   The duration is trough -> peak (``'refine'``: of the refined positions, as in
   v1.0.0; ``'band'``: of the band positions), with both extremes located to a
   fraction of a sample by a parabola through the three samples around them; with
   ``'paper'`` it is the negative half-wave between its two interpolated zero
   crossings (the paper's "zero-crossings separated by 1.1-2 s" for 0.5-0.9 Hz). This
   gate (more than the filter) defines the *effective* band of the detected waves,
   and it is the same at every ``fs``: a sine at exactly ``fband[1]`` is detected, one
   3 % above it is not, at 128 Hz as at 5 kHz (v1.0.0 lost about a third of the
   ``fband[1]`` sines; the previous one-sample tolerance let more out-of-band waves in
   at low ``fs``).
6. **Edges and gaps.** A wave's *span* runs from its interpolated down-going zero
   crossing (``zero_pos_frac``) to the end of its positive half-wave (``end_pos``),
   widened to include the trough and peak if they lie outside (possible with
   ``'refine'``). A wave is discarded when its span starts within ``edge_margin_s`` of
   the first sample or ends within ``edge_margin_s`` of the last sample, or overlaps a
   gap widened by its *gap margin* on each side. See *Edges and gaps*.
7. **Morphology.** Amplitudes, durations and slopes are read at the wave's positions
   on both signals (outputs below).

Trough placement (``trough``)
-----------------------------

``'refine'`` (default) -- **the v1.0.0 placement**
    The trough / peak are the extremes of the drift-removed ``x`` within +/- half a
    period of ``fband[1]`` around the band-pass extremes. As in v1.0.0 this window is
    not clamped to the wave's half-wave, the duration gate uses these refined
    positions, the broadband values are read from ``measure_on`` (when given) at these
    positions, and ``downslope`` runs from the first negative sample of the band
    signal (``zero_pos``), not the interpolated crossing. A trough that lands at or
    before that sample gives ``downslope = NaN``, which the slope features ignore.
    This keeps existing results: on the demo recording (6.8 h Fz-Cz, 500 Hz, NREM
    epochs, ``measure_on`` 0.5-35 Hz, ``amplitude_threshold=5``), the mean downslope
    is SO 186.8 / delta 250.1 uV/s with ``filter='fft'`` (v1.0.0: 186.0 / 250.2; the
    rest of the difference is the truncated edge half-waves no longer being paired and
    the fs-independent gate tolerance) and **173.9 / 239.3** with the default
    Butterworth filter (the ringing fix, -6.5 % / -4.4 %). The search runs on ``x``
    even when ``measure_on`` is given (as in v1.0.0); use ``'paper'`` to measure the
    negative peak of the measured trace itself.
    Why these v1.0.0 details matter: clamping the window to the half-wave, or using
    the interpolated crossing, lets broadband troughs sit a fraction of a sample after
    the crossing, and ``-min_val / down_dur`` then explodes (demo SO mean 316, with
    ``measure_on`` as the search signal 910). In v1.0.0 these cases gave NaN.
``'band'`` -- band-limited placement (opt-in)
    The trough / peak are those of the band-passed signal; the broadband values are
    read there. Unbiased under noise (100 uV pk2pk 1 Hz sine + white noise SD 10 uV:
    pk2pk 101 vs 139 with ``'refine'``) and no heavy slope tail, but it reads the
    broadband trace at the band trough, which is neither the broadband negative peak
    nor the band slope, and sharp non-sinusoidal extremes are underestimated. Demo:
    SO 51.8 / delta 155.0 uV/s (mean), rate 0.309 / 1.336 per s.
``'paper'`` -- Carvalho et al. 2024, Methods (opt-in)
    The detector builds the study's trace from ``x``: a 0.5-35 Hz zero-phase FIR
    (Hamming window, 4 s; 1999 taps at the study's 500 Hz, applied once with its delay
    removed, DC gain set to exactly 0), then a centred 50 ms moving average (25 samples
    at 500 Hz). Zero crossings, half-waves, the duration gate (step 5) and the trough
    -- "the negative peak after each zero-crossing" -- are all taken on that trace, and
    the slope is "the amplitude difference (zero-crossing to negative peak) divided by
    their interval" on the same trace. ``fband`` only sets the gate; ``measure_on``
    must be None and ``features_on`` must be ``'broadband'``. The ``_band`` outputs are
    the band-passed ``x`` read at the same positions. Default ``edge_margin_s`` and the
    gap-margin cap are the support of the two filters (2.02 s at 500 Hz): beyond it the
    fill or the signal end has exactly no effect. Wave for wave identical to an
    independent loop implementation of the Methods text on the demo recording. Demo:
    SO 173.1 / delta 195.3 uV/s (mean of epoch means), median wave 104.2 / 152.0;
    only 1.3 SO waves per NREM epoch (half-waves of 0.55-1 s are rare on a 0.5-35 Hz
    trace). The paper reports 95.1 +/- 28.9 (SO) and 130.8 +/- 34.8 (delta) across
    subjects; this is one subject, and the original ``SlowWaveDetect`` source is not
    available, so this mode reproduces the published *method*, not verified numbers.

Edges and gaps
--------------

Both margins exist because a zero-phase filter needs time to forget a signal end or a
gap fill. They were measured on 1/f EEG with slow oscillations and delta bursts (fs 250
and 1000 Hz; bands 0.5-4, 0.5-0.9 and 1-3.9 Hz; ``'refine'`` and ``'band'``; probes
listed in PR #68) by comparing every wave with the same wave in the uncut, gap-free
recording, against the distance between the wave's span and the cut or gap, in periods
of ``fband[0]``: *value error* = broadband trough or peak value changed by more than
1 uV; *lost / gained* = no wave with a zero crossing within 20 ms; *switched* = trough
or peak moved by more than 2 samples (1 nV of white noise alone switches 0.1-1 %, between
near-equal extremes).

* ``edge_margin_s`` (default ``4 / fband[0]``, 8 s for a 0.5 Hz edge; ``'paper'``: the
  trace's filter support). At 3-3.5 periods from a signal end 1.5 % of waves still had a
  value error (0.5-4 Hz), none beyond 3.5 periods. Use ``context`` for consecutive
  segments so this costs nothing at the segment borders.
* gap margin (default: depends on the gap length ``L``)::

      margin = 0                                        if L <= 25 ms
      margin = min(4, 4 * (L * fband[0]) ** 0.25) / fband[0]   otherwise

  =====================  ========  =====  =====  =====  =====  ======
  gap length ``L``       <= 25 ms  50 ms  0.1 s  0.5 s  1 s    >= 2 s
  margin at 0.5 Hz (s)   0         3.2    3.8    5.7    6.7    8.0
  =====================  ========  =====  =====  =====  =====  ======

  With this rule (brainmaze-utils' spectral fill, utils#26 round 3), among the waves
  within 6 periods of gaps of 4 ms to 5 s, at most 0.8 % had a value error (0.5-0.9 Hz
  band; 0 % in the others), 0.1 % were lost or gained and 0.5 % switched; the same at
  1 kHz. Gaps up to 20 ms changed nothing even with no margin (a 1-sample dropout costs
  only the wave it falls in). Without margins long gaps gave up to 18 % value errors;
  the previous fixed ``3 / fband[0]`` up to 1.1 %. With
  ``'paper'`` the margin is capped at the trace's filter support. A fixed
  ``gap_margin_s`` (seconds, all gaps) overrides the rule.

``WAVE_RATE`` counts waves per *analysable* second: the time outside the margins and
gaps, with every clean run shortened by the mean zero-crossing -> trough time at its
start and the mean trough -> span-end time at its end -- the same rule that decides
whether a wave is kept. So the rate does not depend on how many gaps there are (1 Hz
sine with 20 ms dropouts every 10 s or 50 ms every 30 s: pooled rate 1.000 within 1 %;
before this, -6 to -11 %). ``ANALYSABLE_RATE`` (with ``datarate=True``) is that time as a
fraction of the window: gate ``WAVE_RATE`` on it, not on ``DATA_RATE``.

Two signals: filtered and unfiltered
------------------------------------

Every wave is measured twice, on the same set of detected waves:

* **unfiltered** (broadband; keys without suffix, features ``WAVE_*``): values of the
  drift-removed input (or of ``measure_on``, or the paper trace) -- what the EEG
  really looked like, including faster activity riding on the wave;
* **filtered** (band-passed; keys suffixed ``_band``, features suffixed ``_BAND``):
  values of the band-passed detection signal at its own extremes -- the band-limited
  component only. Note the filter gain: ``|H(f)|^2`` is 0.5 at the band edges, so the
  filtered amplitude of a wave near a band edge is attenuated (a pure sine at
  ``fband[0]`` or ``fband[1]`` comes out at half amplitude).

``WaveDetector(features_on=...)`` selects which set the windowed features use;
``detect()`` always returns both.

Outputs (per wave, from ``detect`` / :func:`detect_waves`)
----------------------------------------------------------

``zero_pos``
    First negative sample after the down-going zero crossing (int sample index).
``zero_pos_frac``
    The same crossing linearly interpolated between the two samples that bracket it
    (fractional sample index).
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
    ``(min_pos - zero) / fs`` (s), where ``zero`` is ``zero_pos`` for the broadband
    outputs of ``trough='refine'`` (v1.0.0) and ``zero_pos_frac`` otherwise. Can be
    <= 0 with ``'refine'``.
``upslope``
    ``pk2pk / delta_t`` (input units / s).
``downslope``
    ``-min_val / down_dur`` (input units / s); NaN when ``down_dur <= 0``. Positive for
    a trough below zero (the normal case). It can be negative if the unfiltered value
    at the trough is above zero; ``amplitude_threshold`` removes such waves.
``*_band``
    The nine keys ``min_pos`` ... ``downslope`` again, measured on the filtered signal
    (always with the interpolated crossing).
``gaps``
    ``(n_gaps, 2)`` ``[start, stop)`` sample indices of the non-finite runs.
``x_band``, ``x_amp``
    Only with ``return_signals=True``: the filtered and unfiltered signals (NaN in
    gaps).

Windowed features (``__call__``)
--------------------------------

See :class:`WaveDetector`. In order: ``DATA_RATE`` (only with ``datarate=True``),
``WAVE_RATE``, the per-window means of ``pk2pk``, the selected slope, ``delta_t``,
``min_val`` and ``max_val`` (the v1.0.0 names, at their v1.0.0 positions), then
``WAVE_SLOPE_MEDIAN`` (median of the selected slope: robust to the few very steep
waves that dominate the mean), the same for ``_BAND`` when requested, and last
``ANALYSABLE_RATE`` (only with ``datarate=True``).

``DATA_RATE`` here is the fraction of samples that are finite in ``x`` **and** in
``measure_on``; ``TimeDomainFeatureExtractor``'s ``DATA_RATE`` counts only NaN in its
own input. They agree for NaN-only gaps without ``measure_on``; with +/-inf samples or
gaps in ``measure_on`` this one is lower.

Slope conventions
-----------------

``slope='downslope'`` (default)
    Zero-crossing -> negative-trough rate, ``-min_val / (t_trough - t_zero_cross)``,
    the slow-wave slope of Carvalho et al. 2024. The study detected on the narrow band,
    measured on a broadband 0.5-35 Hz trace and kept waves with a negative peak of at
    least 5 uV (``amplitude_threshold=5``); ``trough='paper'`` implements its Methods
    text, the default reproduces this package's v1.0.0 numbers. See
    ``demo/eeg_wave_detection/example_one_file.py``.
``slope='upslope'``
    Trough -> peak rate, ``(max_val - min_val) / (t_peak - t_trough)``.

For ``A sin(2 pi f t)`` both equal ``4 A f``.

Performance
-----------

Everything is vectorised (no Python loop over samples, waves or windows); time and
memory are O(n) per signal, dominated by the ``sosfiltfilt`` passes. 2-D input is
converted to float64 one channel at a time; the working set is about five float64
copies of one channel (~1.8 GB for a 30-min channel at 25 kHz), times ``n_processes``.
CPU time per 30-minute channel (single thread, ``fband=(0.5, 4)``, ``segm_size=30``,
synthetic 1/f EEG with delta bursts, on a loaded 12-core workstation): 0.054 s at
250 Hz, 0.20 s at 1 kHz (0.32 s with 20 gaps), 1.0 s at 5 kHz, 5.3 s at 25 kHz;
``trough='paper'`` adds 10-45 % (its FIR). v1.0.0 needed 0.40 / 1.56 / 8.7 / 48.4 s
(7.5-9x slower) and returned ``WAVE_RATE = 0`` for any input with a gap.

Changes from v1.0.0 (class version 2.0.0 -> 2.1.0)
--------------------------------------------------

Same default trough placement as v1.0.0 (``trough='refine'``). Butterworth instead of
brick-wall FFT filter (demo SO / delta downslope 186.0 / 250.2 -> 173.9 / 239.3 uV/s);
NaN gaps no longer zero the whole signal; truncated edge half-waves are no longer
paired; waves near the signal edges and gaps are excluded and ``WAVE_RATE`` is
normalised by analysable time; fs-independent duration gate; ``trough='band'`` and
``trough='paper'``; ``WAVE_SLOPE_MEDIAN``, ``ANALYSABLE_RATE``, ``*_band`` outputs and
``features_on``; ``return_signals``; ``context``; numpy scalar ``fs`` accepted.

References
----------
Carvalho D.Z. et al. (2024), *Non-rapid eye movement sleep slow-wave activity features
are associated with amyloid accumulation in older adults with obstructive sleep apnoea*,
Brain Communications 6(5): fcae354. https://doi.org/10.1093/braincomms/fcae354.
Methods, slow-wave detection: "bandpass finite impulse response filter with zero phase
shift ... in the 0.5-35 Hz band" (length 2000, Hamming window), then a "50-ms moving
average filter"; "negative peaks (troughs) of SWs after each zero-crossing";
"zero-crossings were separated by 1.1-2 s for SO and 0.25-1.0 s for delta waves";
"SW amplitude threshold set at -5 uV"; slope = "the amplitude difference (zero-crossing
to negative peak in uV) by their interval".

Lineage: this detector is the successor of the ``SlowWaveDetect`` routine used in the
study above, generalised to an arbitrary band. The original ``SlowWaveDetect`` source is
not part of this repository, so numerical identity with the published values cannot be
verified here.
"""

import multiprocessing
import numbers
import warnings
from functools import partial

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, firwin, oaconvolve, sosfiltfilt

from brainmaze_utils.gaps import fill_gaps, find_gaps, mask_in_gaps

__all__ = ['WaveDetector', 'detect_waves']

#: per-wave keys of a detection dict (all arrays of length n_waves)
_MORPH_KEYS = ('min_pos', 'min_val', 'max_pos', 'max_val', 'pk2pk', 'delta_t',
               'upslope', 'down_dur', 'downslope')
_WAVE_KEYS = (('zero_pos', 'zero_pos_frac', 'end_pos') + _MORPH_KEYS
              + tuple(k + '_band' for k in _MORPH_KEYS))
_INT_KEYS = ('zero_pos', 'end_pos', 'min_pos', 'max_pos', 'min_pos_band', 'max_pos_band')
#: keys holding sample positions (shifted when ``context`` is used)
_POS_KEYS = ('zero_pos', 'zero_pos_frac', 'end_pos', 'min_pos', 'max_pos',
             'min_pos_band', 'max_pos_band')

_FILTERS = ('butter', 'fft')
_NAN_POLICIES = ('fill', 'raise')
#: trough / peak placement modes (module docstring, *Trough placement*)
_TROUGH_MODES = ('refine', 'band', 'paper')

#: drift removal for filter='butter': Butterworth high-pass of this order at
#: _DRIFT_FACTOR * fband[0], forward-backward. Gain at fband[0] = 1/(1 + 0.5**8) = 0.996,
#: so in-band amplitudes are preserved to < 0.4 %; gain at fband[0]/4 = 0.004.
_DRIFT_ORDER = 4
_DRIFT_FACTOR = 0.5

#: trough='refine': search half-window in periods of fband[1] (v1.0.0: half a period)
_REFINE_HALF_WIN = 0.5

#: relative tolerance of the duration gate (sub-sample durations, independent of fs)
_GATE_RTOL = 0.01

#: gap fill passed explicitly to brainmaze_utils.gaps.fill_gaps
_FILL_METHOD = 'spectral'
_FILL_MAX_INTERP_S = 0.1

#: default edge margin, in periods of fband[0] (module docstring, *Edges and gaps*)
_EDGE_MARGIN_PERIODS = 4.0
#: default gap margin (s) = 0 for gaps <= _GAP_MARGIN_SHORT_S, else
#: min(_GAP_MARGIN_MAX_PERIODS, _GAP_MARGIN_SLOPE * (L * f_low) ** _GAP_MARGIN_POWER) / f_low
#: (see _default_gap_margins and the module docstring, *Edges and gaps*)
_GAP_MARGIN_MAX_PERIODS = 4.0
_GAP_MARGIN_SHORT_S = 0.025
_GAP_MARGIN_SLOPE = 4.0
_GAP_MARGIN_POWER = 0.25

#: trough='paper' (Carvalho et al. 2024, Methods): 0.5-35 Hz zero-phase FIR (Hamming
#: window, ~4 s = 2000 taps at the study's 500 Hz), then a 50 ms moving average
_PAPER_BAND = (0.5, 35.0)
_PAPER_FIR_S = 4.0
_PAPER_MA_S = 0.05


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


def _band_signal(x0, fs, f_low, f_high, filter, order):
    """Detection signal: ``x0`` band-passed to ``(f_low, f_high)`` (module docstring, step 2)."""
    if filter == 'fft':
        return _bandpass_fft(x0, fs, f_low, f_high)
    sos_bp = butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    return _sosfiltfilt(sos_bp, x0, fs, f_low)


def _drift_removed(x0, fs, f_low, filter):
    """
    Drift-removed ``x0``: 4th-order Butterworth high-pass at ``0.5 * f_low``
    (forward-backward) for ``filter='butter'``, brick-wall high-pass at ``f_low`` for
    ``'fft'`` (the v1.0.0 reference).
    """
    if filter == 'fft':
        return x0 - _bandpass_fft(x0, fs, 0.0, f_low)
    f_d = _DRIFT_FACTOR * f_low
    sos_hp = butter(_DRIFT_ORDER, f_d, btype='highpass', fs=fs, output='sos')
    return _sosfiltfilt(sos_hp, x0, fs, f_d)


def _paper_kernel_sizes(fs):
    """``(numtaps, ma_len)`` of the ``trough='paper'`` trace: both odd (zero phase)."""
    numtaps = 2 * int(round(_PAPER_FIR_S / 2.0 * fs)) - 1       # 1999 at 500 Hz
    ma_len = 2 * int(round(_PAPER_MA_S / 2.0 * fs)) + 1         # 25 at 500 Hz (50 ms)
    return numtaps, ma_len


def _paper_support_s(fs):
    """
    Half-length (s) of the ``trough='paper'`` trace's impulse response: samples further
    than this from a gap or a signal end do not depend on the fill or the padding at all.
    """
    numtaps, ma_len = _paper_kernel_sizes(fs)
    return (numtaps // 2 + ma_len // 2) / fs


def _paper_taps(fs):
    """
    The paper's FIR: ``firwin(numtaps, [0.5, 35], window='hamming')``, with its DC gain
    (-0.004 for this design) removed by subtracting a Hamming-shaped correction, so the
    trace does not depend on the signal's (segment's) mean. Symmetric -> linear phase.
    """
    numtaps, _ = _paper_kernel_sizes(fs)
    taps = firwin(numtaps, list(_PAPER_BAND), pass_zero='bandpass', window='hamming', fs=fs)
    w = np.hamming(numtaps)
    return taps - w * (taps.sum() / w.sum())


def _paper_trace(x0, fs):
    """
    The measured trace of Carvalho et al. 2024: ``x0`` band-passed 0.5-35 Hz by a
    Hamming-window FIR (``numtaps`` ~ 4 s, odd, DC gain zeroed; applied once with its
    group delay removed, i.e. linear phase -> zero phase; magnitude = the FIR design),
    then a centred 50 ms moving average.
    """
    numtaps, ma_len = _paper_kernel_sizes(fs)
    taps = _paper_taps(fs)
    y = oaconvolve(x0, taps, mode='same')
    return uniform_filter1d(y, ma_len, mode='nearest')


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


def _parabolic_offset(z, pos):
    """
    Sub-sample offset (in ``[-0.5, 0.5]``) of the extreme at ``pos`` from a parabola
    through ``z[pos-1], z[pos], z[pos+1]``. ``pos`` must be interior.
    """
    y0, y1, y2 = z[pos - 1], z[pos], z[pos + 1]
    den = y0 - 2.0 * y1 + y2
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where(den != 0, 0.5 * (y0 - y2) / den, 0.0)
    return np.clip(off, -0.5, 0.5)


def _half_waves(z):
    """
    Split ``z`` at its zero crossings and pair every negative half-wave with the
    positive half-wave that follows it. Both must be bounded by **real** zero crossings
    (the truncated first and last segments are never paired).

    Returns ``None`` (no pair) or ``(seg_starts, seg_ends, neg_idx)``: segment bounds
    (``seg_ends`` inclusive) and the indices of the paired negative segments.
    """
    n = z.shape[0]
    if n < 3:
        return None
    s = _forward_fill_sign(z)
    change = np.flatnonzero(s[1:] != s[:-1]) + 1            # first sample of each segment
    if change.size == 0:
        return None
    seg_starts = np.concatenate(([0], change))
    seg_ends = np.concatenate((change, [n])) - 1             # inclusive
    n_seg = seg_starts.size
    seg_polarity = s[seg_starts]                            # +/-1 per segment
    pair = (seg_polarity[:-1] < 0) & (seg_polarity[1:] > 0)
    pair[0] = False                                         # segment 0 starts at the border
    pair[n_seg - 2] = False                                 # last segment ends at the border
    neg_idx = np.flatnonzero(pair)
    if neg_idx.size == 0:
        return None
    return seg_starts, seg_ends, neg_idx


def _down_crossing(z, zero_pos):
    """Linearly interpolated position of the positive->negative crossing before ``zero_pos``."""
    a = z[zero_pos - 1]                                     # >= 0
    b = z[zero_pos]                                         # < 0
    return (zero_pos - 1) + a / (a - b)


def _up_crossing(z, first_pos):
    """Linearly interpolated position of the negative->positive crossing before ``first_pos``."""
    c = z[first_pos - 1]                                    # <= 0
    d = z[first_pos]                                        # > 0
    return (first_pos - 1) + (-c) / (d - c)


def _gate(dur, fs, f_low, f_high):
    """Duration gate in samples (module docstring, step 5): fs-independent tolerance."""
    lo = fs / (2.0 * f_high) * (1.0 - _GATE_RTOL)
    hi = fs / (2.0 * f_low) * (1.0 + _GATE_RTOL)
    return (dur >= lo) & (dur <= hi)


def _empty_pairs():
    e = np.empty(0, dtype=np.int64)
    return (e, e.copy(), e.copy(), np.empty(0, dtype=np.float64), e.copy(), e.copy(), e.copy())


def _find_wave_pairs(x_narrow, fs, f_low, f_high):
    """
    Detect waves on the band-passed signal with the band positions (``trough='band'``).

    See steps 3-5 of the module docstring. Fully vectorised, O(n) time and memory.

    Returns
    -------
    trough_pos, peak_pos : np.ndarray[int]
        Broadband trough / peak positions (here: the band positions).
    zero_pos : np.ndarray[int]
        First negative sample after the positive->negative crossing.
    zero_frac : np.ndarray[float]
        Linearly interpolated position of that crossing.
    band_trough, band_peak : np.ndarray[int]
        Trough / peak positions on ``x_narrow``.
    end_pos : np.ndarray[int]
        Last sample of the positive half-wave. ``[zero_frac, end_pos]`` is the span.
    """
    hw = _half_waves(x_narrow)
    if hw is None:
        return _empty_pairs()
    seg_starts, seg_ends, neg_idx = hw

    trough_pos = _argext_groups(x_narrow, seg_starts, 'min')[neg_idx]
    peak_pos = _argext_groups(x_narrow, seg_starts, 'max')[neg_idx + 1]

    # duration gate on sub-sample (parabolic) band extremes, relative tolerance
    dur = ((peak_pos + _parabolic_offset(x_narrow, peak_pos))
           - (trough_pos + _parabolic_offset(x_narrow, trough_pos)))
    keep = _gate(dur, fs, f_low, f_high)
    neg_idx = neg_idx[keep]
    band_trough, band_peak = trough_pos[keep], peak_pos[keep]
    zero_pos = seg_starts[neg_idx]
    zero_frac = _down_crossing(x_narrow, zero_pos)
    end_pos = seg_ends[neg_idx + 1]

    return (band_trough.copy(), band_peak.copy(), zero_pos, zero_frac,
            band_trough, band_peak, end_pos)


def _find_refined_waves(x_narrow, x_ref, fs, f_low, f_high):
    """
    ``trough='refine'`` (the v1.0.0 placement): every band half-wave pair, trough / peak
    moved to the extreme of ``x_ref`` within +/- half a period of ``f_high`` around the
    band extreme (**not** clamped to the half-wave, as in v1.0.0), and the duration gate
    applied to these refined positions (sub-sample, relative tolerance). Same return
    layout as :func:`_find_wave_pairs`.
    """
    hw = _half_waves(x_narrow)
    if hw is None:
        return _empty_pairs()
    seg_starts, seg_ends, neg_idx = hw
    band_trough = _argext_groups(x_narrow, seg_starts, 'min')[neg_idx]
    band_peak = _argext_groups(x_narrow, seg_starts, 'max')[neg_idx + 1]
    half_win = int(round(_REFINE_HALF_WIN * fs / f_high))
    n = x_narrow.shape[0]
    trough_pos = _refine_unclamped(x_ref, band_trough, half_win, 'min')
    peak_pos = _refine_unclamped(x_ref, band_peak, half_win, 'max')
    dur = ((peak_pos + _parabolic_offset_safe(x_ref, peak_pos))
           - (trough_pos + _parabolic_offset_safe(x_ref, trough_pos)))
    keep = _gate(dur, fs, f_low, f_high)
    neg_idx = neg_idx[keep]
    zero_pos = seg_starts[neg_idx]
    return (trough_pos[keep], peak_pos[keep], zero_pos, _down_crossing(x_narrow, zero_pos),
            band_trough[keep], band_peak[keep], seg_ends[neg_idx + 1])


def _refine_unclamped(x_ref, positions, half_win, want, chunk=1 << 22):
    """
    Extreme of ``x_ref`` within ``[pos - half_win, pos + half_win]`` (clipped to the
    signal; ties -> first). Windows of neighbouring waves may overlap. The values are
    gathered through a strided view in chunks of at most ``chunk`` elements; total work
    is n_waves * (2 * half_win + 1), which is ~n for a band-limited detection signal
    (about one pair per period of the band, window ~ one period of ``fband[1]``).
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0 or half_win <= 0:
        return positions
    n = x_ref.shape[0]
    fill = np.inf if want == 'min' else -np.inf             # padding never wins
    xp = np.concatenate((np.full(half_win, fill), x_ref, np.full(half_win, fill)))
    view = np.lib.stride_tricks.sliding_window_view(xp, 2 * half_win + 1)
    out = np.empty_like(positions)
    step = max(1, chunk // (2 * half_win + 1))
    for i in range(0, positions.size, step):
        p = positions[i:i + step]
        win = view[p]
        rel = win.argmin(axis=1) if want == 'min' else win.argmax(axis=1)
        out[i:i + step] = p - half_win + rel
    return np.clip(out, 0, n - 1)


def _parabolic_offset_safe(z, pos):
    """:func:`_parabolic_offset` that returns 0 at the first / last sample."""
    pos = np.asarray(pos, dtype=np.int64)
    inner = (pos > 0) & (pos < z.shape[0] - 1)
    off = np.zeros(pos.shape)
    if inner.any():
        off[inner] = _parabolic_offset(z, pos[inner])
    return off


def _find_paper_waves(trace, fs, f_low, f_high):
    """
    ``trough='paper'``: zero crossings, half-waves and extremes all on the measured
    trace (Carvalho et al. 2024). A negative half-wave is kept when its duration between
    the interpolated down- and up-going zero crossings is within half a period of the
    band edges (the paper's "zero-crossings separated by 1.1-2 s" for 0.5-0.9 Hz).
    Same return layout as :func:`_find_wave_pairs`; the band positions equal the
    broadband ones.
    """
    hw = _half_waves(trace)
    if hw is None:
        return _empty_pairs()
    seg_starts, seg_ends, neg_idx = hw
    zero_frac = _down_crossing(trace, seg_starts[neg_idx])
    up_frac = _up_crossing(trace, seg_starts[neg_idx + 1])
    keep = _gate(up_frac - zero_frac, fs, f_low, f_high)
    neg_idx, zero_frac = neg_idx[keep], zero_frac[keep]
    trough_pos = _argext_groups(trace, seg_starts, 'min')[neg_idx]
    peak_pos = _argext_groups(trace, seg_starts, 'max')[neg_idx + 1]
    zero_pos = seg_starts[neg_idx]
    end_pos = seg_ends[neg_idx + 1]
    return trough_pos, peak_pos, zero_pos, zero_frac, trough_pos.copy(), peak_pos.copy(), end_pos


def _empty_detection():
    d = {k: np.empty(0, dtype=np.float64) for k in _WAVE_KEYS}
    for k in _INT_KEYS:
        d[k] = np.empty(0, dtype=np.int64)
    return d


def _check_margin(margin_s, name):
    """``None`` stays ``None`` (default rule); otherwise a non-negative number of seconds."""
    if margin_s is None:
        return None
    if isinstance(margin_s, bool) or not isinstance(margin_s, numbers.Real) \
            or not np.isfinite(margin_s) or margin_s < 0:
        raise ValueError(f'{name} must be None or a non-negative number of seconds. '
                         f'Got: {margin_s!r}')
    return float(margin_s)


def _check_trough(trough):
    if not isinstance(trough, str) or trough not in _TROUGH_MODES:
        raise ValueError(f'trough must be one of {_TROUGH_MODES}. Got: {trough!r}')
    return trough


def _default_edge_margin(f_low, fs, trough):
    """Default ``edge_margin_s`` (module docstring, *Edges and gaps*)."""
    if trough == 'paper':
        return _paper_support_s(fs)
    return _EDGE_MARGIN_PERIODS / f_low


def _default_gap_margins(gap_len_s, f_low, fs, trough):
    """
    Default per-gap margin (s) as a function of the gap length ``L`` (s)::

        margin = min(4, 4 * (L * f_low) ** 0.25) / f_low   if L > 0.025 s, else 0

    i.e. in periods of ``f_low`` (at 0.5 Hz): 0 for gaps up to 25 ms, 1.6 for 50 ms, 1.9
    for 0.1 s, 2.8 for 0.5 s, 3.4 for 1 s and 4 for gaps of ``2 / f_low`` and longer. With ``trough='paper'`` it
    is capped at the support of the paper trace's filters (beyond which the fill has
    exactly no effect). Measurements: module docstring, *Edges and gaps*.
    """
    L = np.asarray(gap_len_s, dtype=np.float64)
    periods = np.minimum(_GAP_MARGIN_MAX_PERIODS,
                         _GAP_MARGIN_SLOPE * (np.maximum(L, 0.0) * f_low) ** _GAP_MARGIN_POWER)
    m = np.where(L > _GAP_MARGIN_SHORT_S, periods / f_low, 0.0)
    if trough == 'paper':
        m = np.minimum(m, _paper_support_s(fs))
    return m


def _gap_margins(gaps, fs, f_low, gap_margin_s, trough):
    """Per-gap margins in seconds (fixed ``gap_margin_s``, or the default rule)."""
    gaps = np.asarray(gaps).reshape(-1, 2)
    if gap_margin_s is not None:
        return np.full(gaps.shape[0], float(gap_margin_s))
    return _default_gap_margins((gaps[:, 1] - gaps[:, 0]) / fs, f_low, fs, trough)


def _excluded_mask(n, gaps, fs, gap_margins_s, edge_margin_s):
    """
    Boolean mask (length ``n``) of samples that are *not analysable*: inside a gap
    widened by its margin on each side, or within ``edge_margin_s`` of either end of the
    signal.

    Uses exactly the inequalities of the per-wave exclusion in :func:`detect_waves` (a
    sample at time ``t = i / fs`` is excluded iff ``lo <= t < hi`` for a widened gap
    ``[lo, hi)``, or ``t < edge_margin_s``, or ``t > (n - 1) / fs - edge_margin_s``).
    """
    t = np.arange(n) / fs
    bad = (t < edge_margin_s) | (t > (n - 1) / fs - edge_margin_s)
    g = np.asarray(gaps, dtype=np.float64).reshape(-1, 2)
    if g.size:
        m = np.asarray(gap_margins_s, dtype=np.float64)
        lo = np.ceil((g[:, 0] / fs - m) * fs).astype(np.int64)
        hi = np.ceil((g[:, 1] / fs + m) * fs).astype(np.int64)
        lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
        delta = np.zeros(n + 1, dtype=np.int64)
        np.add.at(delta, lo, 1)
        np.add.at(delta, hi, -1)
        bad |= np.cumsum(delta[:-1]) > 0
    return bad


def _shrink_runs(ok, before, after):
    """
    Shrink every run of True in ``ok`` by ``before`` samples at its start and ``after``
    at its end (runs shorter than ``before + after`` vanish). Returns a new mask.
    """
    if before <= 0 and after <= 0:
        return ok.copy()
    d = np.diff(np.concatenate(([0], ok.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)                          # exclusive
    s2, e2 = starts + before, ends - after
    keep = e2 > s2
    s2, e2 = s2[keep], e2[keep]
    n = ok.size
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, s2, 1)
    np.add.at(delta, e2, -1)
    return np.cumsum(delta[:-1]) > 0


def detect_waves(x, fs, fband=(0.5, 4.0), measure_on=None, filter='butter',
                 filter_order=2, nan_policy='fill', gap_margin_s=None,
                 edge_margin_s=None, trough='refine', return_signals=False):
    """
    Detect waves in a single 1-D signal and return their positions and morphology.

    Parameters
    ----------
    x : np.ndarray
        1-D signal (any amplitude unit; outputs use the same unit, e.g. uV). Any real
        dtype; it is converted to float64 here (one signal at a time).
    fs : float
        Sampling frequency (Hz). Python or numpy real scalar.
    fband : (float, float)
        ``(low, high)`` band in Hz, ``0 < low < high < fs/2``.
    measure_on : np.ndarray, optional
        Signal the *broadband* amplitudes/slopes are read from (same length as ``x``),
        e.g. a 0.5-35 Hz trace of ``x``. Mean-subtracted, otherwise used as-is.
        Detection (zero crossings, half-waves) always runs on ``x``; with
        ``trough='refine'`` the broadband trough / peak are searched on ``measure_on``,
        the trace whose values are reported. Defaults to the drift-removed ``x``. The
        ``*_band`` outputs always come from the band-passed ``x``. Not allowed with
        ``trough='paper'`` (which builds its own trace).
    filter : {'butter', 'fft'}
        Band-pass implementation (module docstring, step 2). Default ``'butter'``.
    filter_order : int
        Butterworth order (ignored for ``'fft'``). The filter is applied
        forward-backward, so the magnitude response is squared. Default 2.
    nan_policy : {'fill', 'raise'}
        Non-finite samples in ``x`` or ``measure_on``: ``'fill'`` (default) fills them
        for filtering and discards every wave that overlaps them (plus the gap margin);
        ``'raise'`` raises ``ValueError``.
    gap_margin_s : float or None
        Seconds added on both sides of every gap; a wave whose span overlaps a widened
        gap is discarded. ``None`` (default): depends on the gap length (0 up to 25 ms,
        rising to ``4 / fband[0]`` for gaps of ``2 / fband[0]`` and longer; module
        docstring, *Edges and gaps*).
    edge_margin_s : float or None
        A wave whose span starts within ``edge_margin_s`` of the first sample or ends
        within it of the last sample is discarded (the zero-phase filters' edge
        transient). ``None`` (default): ``4 / fband[0]`` (with ``trough='paper'``: the
        support of the paper trace's filters, ~2 s). ``0`` keeps every complete wave.
    trough : {'refine', 'band', 'paper'}
        Where the broadband (unsuffixed) trough / peak are placed (module docstring,
        *Trough placement*). ``'refine'`` (default, v1.0.0 placement): the extreme of
        the broadband trace within half a period of ``fband[1]`` around the band-pass
        extreme, inside the wave's own half-wave. ``'band'``: the band-pass extreme.
        ``'paper'``: Carvalho et al. 2024 -- zero crossings, half-waves and the
        negative peak all on a 0.5-35 Hz FIR + 50 ms moving-average trace of ``x``.
    return_signals : bool
        Also return the band-passed signal (``'x_band'``) and the broadband amplitude
        signal (``'x_amp'``; with ``trough='paper'``, the paper trace), both float
        arrays of the input length. Gaps are NaN in both (the fill is never returned).
        Default False.

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
    trough = _check_trough(trough)
    if trough == 'paper':
        _check_paper_fs(fs)
        if measure_on is not None:
            raise ValueError("trough='paper' measures on its own 0.5-35 Hz + 50 ms trace "
                             "of x; measure_on must be None.")
    gap_margin_s = _check_margin(gap_margin_s, 'gap_margin_s')
    edge_margin_s = _check_margin(edge_margin_s, 'edge_margin_s')
    if edge_margin_s is None:
        edge_margin_s = _default_edge_margin(f_low, fs, trough)

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
    xf = fill_gaps(np.where(invalid, np.nan, x), fs, method=_FILL_METHOD,
                   max_interp_s=_FILL_MAX_INTERP_S) if has_gap else x
    x0 = xf - xf[~invalid].mean()
    del xf

    x_narrow = _band_signal(x0, fs, f_low, f_high, filter, filter_order)
    if trough == 'paper':
        amp = _paper_trace(x0, fs)
        found = _find_paper_waves(amp, fs, f_low, f_high)
    else:
        amp = (_drift_removed(x0, fs, f_low, filter) if amp_raw is None
               else amp_raw - amp_raw[~invalid].mean())
        if trough == 'band':
            found = _find_wave_pairs(x_narrow, fs, f_low, f_high)
        else:
            # v1.0.0: the search runs on the drift-removed x, also when measure_on is given
            x_ref = amp if amp_raw is None else _drift_removed(x0, fs, f_low, filter)
            found = _find_refined_waves(x_narrow, x_ref, fs, f_low, f_high)
            del x_ref
    del x0
    trough_pos, peak_pos, zero_pos, zero_frac, b_trough, b_peak, end_pos = found

    # exclusion: span [zero crossing, end of the positive half-wave] must not touch a
    # widened gap or the edge zones (same inequalities as _excluded_mask)
    if trough_pos.size:
        n = x.shape[0]
        # with trough='refine' the trough / peak can lie outside [zero crossing, end]
        s0 = np.minimum(zero_frac, np.minimum(trough_pos, peak_pos))
        s1 = np.maximum(end_pos, np.maximum(trough_pos, peak_pos))
        bad = (s0 / fs < edge_margin_s) | (s1 / fs > (n - 1) / fs - edge_margin_s)
        if gaps.size:
            m = _gap_margins(gaps, fs, f_low, gap_margin_s, trough)
            wide = np.column_stack((gaps[:, 0] / fs - m, gaps[:, 1] / fs + m))
            bad |= mask_in_gaps(s0 / fs, wide, fs, units='seconds', gap_units='seconds',
                                margin_s=0.0, end=s1 / fs)
        keep = ~bad
        trough_pos, peak_pos, b_trough, b_peak = (trough_pos[keep], peak_pos[keep],
                                                  b_trough[keep], b_peak[keep])
        zero_pos, zero_frac, end_pos = zero_pos[keep], zero_frac[keep], end_pos[keep]

    out.update({'zero_pos': zero_pos, 'zero_pos_frac': zero_frac, 'end_pos': end_pos})
    # v1.0.0 downslope for 'refine': from the first negative sample (zero_pos); a broadband
    # trough at or before it gives NaN (module docstring, *Trough placement*)
    zc_slope = zero_pos.astype(np.float64) if trough == 'refine' else zero_frac
    out.update(_morphology(amp, trough_pos, peak_pos, zc_slope, fs, ''))
    out.update(_morphology(x_narrow, b_trough, b_peak, zero_frac, fs, '_band'))
    if return_signals:
        out['x_band'] = np.where(invalid, np.nan, x_narrow)
        out['x_amp'] = np.where(invalid, np.nan, amp)
    return out


def _check_paper_fs(fs):
    if _PAPER_BAND[1] >= 0.95 * fs / 2:
        raise ValueError(f"trough='paper' band-passes 0.5-35 Hz and needs fs > "
                         f"{2 * _PAPER_BAND[1] / 0.95:.1f} Hz. Got: {fs}")


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


def _check_context(context):
    """``None`` -> ``(0, 0)``; otherwise two non-negative integers (samples)."""
    if context is None:
        return 0, 0
    try:
        nb, na = context
    except (TypeError, ValueError):
        raise ValueError(f'context must be None or (n_before, n_after) samples. Got: {context!r}')
    for v in (nb, na):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
            raise ValueError(f'context must be two non-negative integers (samples). Got: {context!r}')
    return int(nb), int(na)


def _apply_context(det, n, nb, na):
    """Keep waves whose trough lies in the core ``[nb, n - na)`` and shift to core coordinates."""
    if nb == 0 and na == 0:
        return det
    keep = (det['min_pos'] >= nb) & (det['min_pos'] < n - na)
    out = {}
    for k in _WAVE_KEYS:
        v = det[k][keep]
        out[k] = v - nb if k in _POS_KEYS else v
    g = np.asarray(det['gaps']).reshape(-1, 2)
    g = np.clip(g, nb, n - na) - nb
    out['gaps'] = g[g[:, 1] > g[:, 0]].astype(np.int64)
    for k in ('x_band', 'x_amp'):
        if k in det:
            out[k] = det[k][nb:n - na]
    return out


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
        Which slope ``WAVE_SLOPE_MEAN`` / ``WAVE_SLOPE_MEDIAN`` report (see module
        docstring). Default ``'downslope'``.
    amplitude_threshold : float, optional
        Keep only waves whose broadband negative trough is at least this deep
        (``-min_val >= amplitude_threshold``, input units). ``None`` (default) keeps all
        waves. Carvalho et al. use ``5`` (uV). The same set of waves feeds the broadband
        and the band features.
    features_on : {'broadband', 'band', 'both'}
        Which signal the windowed shape features are measured on (module docstring,
        *Two signals*). ``'broadband'`` (default): ``WAVE_*`` names as in v1.0.0.
        ``'band'``: the band-passed signal, names suffixed ``_BAND``. ``'both'``: both
        sets, broadband first. ``DATA_RATE`` and ``WAVE_RATE`` appear once. Must be
        ``'broadband'`` with ``trough='paper'``.
    datarate : bool
        If True, add ``DATA_RATE`` (first; fraction of finite samples per window) and
        ``ANALYSABLE_RATE`` (last; fraction of the window that ``WAVE_RATE`` is
        computed on).
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
        :func:`brainmaze_utils.gaps.fill_gaps` (``method='spectral'``), discard waves
        overlapping a gap (plus its margin), and normalise ``WAVE_RATE`` by the
        analysable time of each window. ``'raise'``: raise ``ValueError``.
    gap_margin_s, edge_margin_s : float or None
        Exclusion margins in seconds around gaps / at the two ends of each signal.
        ``None`` (default): gap-length-dependent / ``4 / fband[0]``. See
        :func:`detect_waves` and the module docstring, *Edges and gaps*.
    trough : {'refine', 'band', 'paper'}
        Where the broadband trough / peak are placed (module docstring, *Trough
        placement*). Default ``'refine'`` (the v1.0.0 placement).

    Notes
    -----
    Windowed features (``__call__``):

    * detection runs once on the **whole** signal (so window borders do not cut waves
      and do not add filter transients); a wave then belongs to the window containing
      its trough (``min_pos``);
    * ``WAVE_RATE`` = number of waves / *analysable* seconds in the window (Hz).
      Analysable: finite, not within the gap margin of a gap or ``edge_margin_s`` of
      either end, and -- because a wave is only kept when its whole span is clean --
      every clean run is shortened by the mean zero-crossing->trough time at its start
      and the mean trough->span-end time at its end (means over the signal's counted
      waves). So the expected rate does not depend on how many gaps there are;
    * a window with no analysable time gives ``WAVE_RATE = NaN``; a window with
      analysable time but no wave gives ``WAVE_RATE = 0``; shape features are NaN when
      there is no wave;
    * ``WAVE_*_MEAN`` are means, ``WAVE_SLOPE_MEDIAN`` the median, over the waves in the
      window, ignoring non-finite per-wave values (e.g. a NaN downslope).
    """

    __version__ = '2.1.0'

    _SHAPE_FEATURES = ('WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN', 'WAVE_DELTA_T_MEAN',
                       'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN', 'WAVE_SLOPE_MEDIAN')
    #: per-wave detection key behind each shape feature (slope resolved at run time)
    _SHAPE_KEYS = {'WAVE_PK2PK_MEAN': 'pk2pk', 'WAVE_SLOPE_MEAN': None,
                   'WAVE_DELTA_T_MEAN': 'delta_t', 'WAVE_MIN_MEAN': 'min_val',
                   'WAVE_MAX_MEAN': 'max_val', 'WAVE_SLOPE_MEDIAN': None}
    _FEATURES_ON = ('broadband', 'band', 'both')

    def __init__(self, fs, fband=(0.5, 4.0), segm_size=None, overlap=0.0,
                 slope='downslope', amplitude_threshold=None,
                 datarate=False, n_processes=1,
                 cutoff_low=None, cutoff_high=None,
                 filter='butter', filter_order=2, nan_policy='fill',
                 gap_margin_s=None, edge_margin_s=None, trough='refine',
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
        trough = _check_trough(trough)
        if trough == 'paper':
            _check_paper_fs(fs)
        if features_on not in self._FEATURES_ON:
            raise ValueError(f'features_on must be one of {self._FEATURES_ON}. Got: {features_on!r}')
        if trough == 'paper' and features_on != 'broadband':
            raise ValueError("trough='paper' defines its waves on the paper trace; the band "
                             "features are not defined there. Use features_on='broadband'.")
        gap_margin = _check_margin(gap_margin_s, 'gap_margin_s')
        edge_margin = _check_margin(edge_margin_s, 'edge_margin_s')

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
        self.trough = trough
        #: fixed gap margin in seconds, or ``None`` for the gap-length-dependent default
        self.gap_margin_s = gap_margin
        #: resolved edge margin in seconds (``None`` at construction -> the default)
        self.edge_margin_s = (edge_margin if edge_margin is not None
                              else _default_edge_margin(f_low, fs, trough))

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
                    edge_margin_s=self.edge_margin_s, trough=self.trough)

    @property
    def feature_names(self):
        """Names returned by ``__call__``, in order (v1.0.0 names keep their positions)."""
        names = (['DATA_RATE'] if self.datarate else []) + ['WAVE_RATE']
        if self.features_on in ('broadband', 'both'):
            names += list(self._SHAPE_FEATURES)
        if self.features_on in ('band', 'both'):
            names += [k + '_BAND' for k in self._SHAPE_FEATURES]
        if self.datarate:
            names.append('ANALYSABLE_RATE')
        return names

    # -- raw detection ------------------------------------------------------------
    def detect(self, x, measure_on=None, return_signals=False, context=None):
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
        context : (int, int), optional
            ``(n_before, n_after)``: the first ``n_before`` and last ``n_after`` samples
            of every signal are context (e.g. the end of the previous and the start of
            the next 30-min segment), used for filtering only. Only waves whose trough
            lies in the core ``[n_before, n - n_after)`` are returned, with all
            positions (and ``gaps``, ``x_band``, ``x_amp``) relative to the core start;
            ``zero_pos`` / ``end_pos`` of a wave that straddles the core border can be
            negative / beyond the core. Consecutive segments then count every wave
            exactly once and lose nothing to the edge margin. Default: no context.

        Returns
        -------
        dict or list of dict
            A single detection dict for 1-D input, otherwise one dict per signal (keys:
            see :func:`detect_waves`; broadband keys unsuffixed, band-passed keys
            suffixed ``_band``). Signals are detected **independently** -- positions
            index into that signal.
        """
        nb, na = _check_context(context)
        signals, measures, single = self._as_signal_list(x, measure_on)
        self._check_context_len(signals, nb, na)
        dets = self._detect_list(signals, measures, False, return_signals)
        dets = [_apply_context(d, s.shape[0], nb, na) for d, s in zip(dets, signals)]
        return dets[0] if single else dets

    def _detect_list(self, signals, measures, single, return_signals=False):
        worker = partial(_detect_one, return_signals=return_signals, **self._detect_kwargs())
        if self.n_processes > 1 and len(signals) > 1:
            with multiprocessing.Pool(min(self.n_processes, len(signals))) as pool:
                results = pool.starmap(worker, list(zip(signals, measures)))
        else:
            results = [worker(s, m) for s, m in zip(signals, measures)]
        return results[0] if single else results

    @staticmethod
    def _check_context_len(signals, nb, na):
        for s in signals:
            if nb + na >= s.shape[0] and (nb or na):
                raise ValueError(f'context ({nb}, {na}) leaves no core samples in a signal '
                                 f'of length {s.shape[0]}.')

    # -- windowed feature extraction ----------------------------------------------
    def __call__(self, x, measure_on=None, context=None):
        """
        Windowed wave features, returned as ``(values, names)`` like the other extractors.

        Parameters
        ----------
        x : np.ndarray or list
            1-D ``(n_samples,)``, 2-D ``(n_channels, n_samples)``, or a list of
            equal-length 1-D arrays.
        measure_on : np.ndarray or list, optional
            Broadband amplitude signal(s), same shape as ``x``.
        context : (int, int), optional
            ``(n_before, n_after)`` context samples at the two ends of every signal,
            used for filtering only (see :meth:`detect`). The feature windows tile the
            core ``[n_before, n - n_after)``.

        Returns
        -------
        values : list of np.ndarray
            One float array per feature; shape ``(n_windows,)`` for 1-D input,
            ``(n_channels, n_windows)`` for 2-D / list input.
        names : list of str
            :attr:`feature_names`: ``[DATA_RATE?, WAVE_RATE, <shape features>,
            ANALYSABLE_RATE?]``. Units: ``DATA_RATE`` and ``ANALYSABLE_RATE`` fractions
            0-1; ``WAVE_RATE`` waves per analysable second (Hz); ``WAVE_PK2PK_MEAN``,
            ``WAVE_MIN_MEAN``, ``WAVE_MAX_MEAN`` input units (e.g. uV);
            ``WAVE_SLOPE_MEAN`` / ``WAVE_SLOPE_MEDIAN`` input units per second;
            ``WAVE_DELTA_T_MEAN`` seconds. ``*_BAND`` features: the same, measured on
            the band-passed signal.
        """
        nb, na = _check_context(context)
        signals, measures, single = self._as_signal_list(x, measure_on)
        self._check_context_len(signals, nb, na)
        detections = self._detect_list(signals, measures, False)

        names = self.feature_names
        per_signal = [self._features_for_signal(sig, m, det, nb, na)
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

    def _rate_mask(self, n, det):
        """
        Samples a counted wave's trough can fall on: analysable samples (finite, outside
        the gap / edge margins), every clean run shortened by the mean zero
        crossing->trough time at its start and the mean trough->span-end time at its end.
        """
        gaps = det['gaps']
        margins = _gap_margins(gaps, self.fs, self.fband[0], self.gap_margin_s, self.trough)
        ok = ~_excluded_mask(n, gaps, self.fs, margins, self.edge_margin_s)
        if det['min_pos'].size:
            before = int(round(float(np.mean(det['min_pos'] - det['zero_pos_frac']))))
            after = int(round(float(np.mean(det['end_pos'] - det['min_pos']))))
            ok = _shrink_runs(ok, before, after)
        return ok

    def _features_for_signal(self, sig, measure, det, nb=0, na=0):
        """All windows of one signal at once (no loop over windows or waves)."""
        n = sig.shape[0]
        starts, ends = self._window_bounds(n - nb - na)
        starts, ends = starts + nb, ends + nb

        finite = np.isfinite(sig)
        if measure is not None:
            finite &= np.isfinite(measure)
        fin_cum = np.concatenate(([0], np.cumsum(finite)))
        n_finite = fin_cum[ends] - fin_cum[starts]

        ana = self._rate_mask(n, det)
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
                row['ANALYSABLE_RATE'] = np.where(ends > starts, n_ana / (ends - starts), np.nan)

        suffixes = []
        if self.features_on in ('broadband', 'both'):
            suffixes.append(('', ''))
        if self.features_on in ('band', 'both'):
            suffixes.append(('_BAND', '_band'))
        for name_sfx, key_sfx in suffixes:
            for feat, key in self._SHAPE_KEYS.items():
                key = (self.slope if key is None else key) + key_sfx
                a = det[key] if order is None else det[key][order]
                agg = _window_nanmedian if feat.endswith('_MEDIAN') else _window_nanmean
                row[feat + name_sfx] = agg(a, lo, hi)
        return row

    def _as_signal_list(self, x, measure_on):
        # no dtype conversion here: each signal is converted to float64 on its own in
        # _detect_one, so a 2-D float32 array is never copied as a whole (R7)
        single = False
        if isinstance(x, np.ndarray) and x.ndim == 1:
            signals = [x]
            single = True
        elif isinstance(x, np.ndarray):
            if x.ndim != 2:
                raise ValueError(f"Input 'x' must be 1-D or 2-D. Got {x.ndim}-D.")
            if x.shape[0] > x.shape[1]:
                warnings.warn(
                    f"WaveDetector: 2-D input has shape {x.shape} -- more signals than "
                    "samples. Input must be (n_signals, n_samples); did you pass "
                    "(n_samples, n_signals)? Transpose it if so.", UserWarning, stacklevel=3)
            signals = list(x)
        elif isinstance(x, (list, tuple)):
            signals = [np.asarray(s).ravel() for s in x]
        else:
            raise ValueError("Input 'x' must be a numpy array or a list of 1-D arrays.")

        if measure_on is None:
            measures = [None] * len(signals)
        elif isinstance(measure_on, np.ndarray) and measure_on.ndim == 1:
            measures = [measure_on]
        elif isinstance(measure_on, np.ndarray):
            measures = list(measure_on)
        else:
            measures = [np.asarray(m) for m in measure_on]
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


def _window_nanmedian(a, lo, hi):
    """
    Median of ``a[lo[k]:hi[k]]`` for every window ``k`` (windows may overlap), ignoring
    non-finite entries; NaN for a window without a finite entry. Vectorised: the window
    slices are gathered once (total length = sum of window wave counts) and sorted by
    ``(window, value)``.
    """
    a = np.asarray(a, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    out = np.full(lo.shape, np.nan)
    lengths = np.maximum(hi - lo, 0)
    total = int(lengths.sum())
    if total == 0:
        return out
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    idx = np.arange(total, dtype=np.int64) + np.repeat(lo - offsets, lengths)
    gid = np.repeat(np.arange(lo.size), lengths)
    v = a[idx]
    f = np.isfinite(v)
    v, gid = v[f], gid[f]
    if v.size == 0:
        return out
    order = np.lexsort((v, gid))
    v, gid = v[order], gid[order]
    cnt = np.bincount(gid, minlength=lo.size)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    has = cnt > 0
    s, c = start[has], cnt[has]
    out[has] = 0.5 * (v[s + (c - 1) // 2] + v[s + c // 2])
    return out


def _detect_one(sig, measure, fs, fband, thr, **kwargs):
    """
    Detect on one signal and apply the optional negative-peak amplitude threshold. The
    float64 conversion happens here, one signal at a time (no up-front copy of a 2-D
    float32 input).
    """
    sig = np.asarray(sig, dtype=np.float64)
    if measure is not None:
        measure = np.asarray(measure, dtype=np.float64)
    det = detect_waves(sig, fs=fs, fband=fband, measure_on=measure, **kwargs)
    if thr is not None and det['min_pos'].size:
        keep = (-det['min_val']) >= thr
        for k in _WAVE_KEYS:
            det[k] = det[k][keep]
    return det
