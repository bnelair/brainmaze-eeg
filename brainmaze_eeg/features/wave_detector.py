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

Slope conventions
-----------------

``slope='upslope'``
    Trough -> peak rate, ``(max_val - min_val) / (t_peak - t_trough)``. This is the
    historical behaviour of this class.
``slope='downslope'``
    Zero-crossing -> negative-trough rate, ``|min_val| / (t_trough - t_zero_cross)``.
    This is the slow-wave downslope used by Carvalho et al. 2024, who measure it on a
    broadband trace (pass that trace via ``measure_on=``) after detecting on the narrow
    band, and apply an amplitude threshold on the negative peak (``amplitude_threshold``).

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
from functools import partial

import numpy as np

__all__ = ['WaveDetector', 'detect_waves']


# ----------------------------------------------------------------------------------
# vectorised core
# ----------------------------------------------------------------------------------
def _bandpass_fft(x, fs, f_low, f_high):
    """
    Ideal (brick-wall) FFT band-pass ``(f_low, f_high]`` in a single FFT/IFFT round trip.

    Replaces the two-pass ``hp -> lp`` composition; both are pure frequency-domain masks.
    Bin frequencies use ``np.fft.fftfreq`` (true ``fs*k/n``), which also handles the
    negative-frequency mirror automatically -- correct for any ``n``, unlike a
    ``linspace(0, fs, n)`` axis (step ``fs/(n-1)``) that is off by one for short signals.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    Xs = np.fft.fft(x)
    freq = np.abs(np.fft.fftfreq(n, d=1.0 / fs))
    mask = (freq > f_low) & (freq <= f_high)
    return np.real(np.fft.ifft(np.where(mask, Xs, 0.0)))


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


def _refine_positions(x_ref, positions, half_win, want):
    """
    Move each detected position to the true extreme of ``x_ref`` within +/- ``half_win``
    samples. Vectorised gather over the (sparse) detected positions.
    """
    if positions.size == 0 or half_win <= 0:
        return positions
    n = x_ref.shape[0]
    lo = np.clip(positions - half_win, 0, n - 1)
    offs = np.arange(2 * half_win + 1)   # symmetric window [pos-half_win, pos+half_win]
    idx = np.clip(lo[:, None] + offs[None, :], 0, n - 1)
    vals = x_ref[idx]
    rel = vals.argmin(axis=1) if want == 'min' else vals.argmax(axis=1)
    return idx[np.arange(idx.shape[0]), rel]


def _find_wave_pairs(x_narrow, x_ref, fs, f_low, f_high):
    """
    Detect (trough, peak, preceding zero-crossing) triples on the band-passed signal.

    A wave is a negative half-wave (its trough) immediately followed by a positive
    half-wave (its peak). Positions are refined on ``x_ref`` and filtered to keep only
    trough->peak durations consistent with the band (half a period of ``f_high`` up to
    half a period of ``f_low``).

    Returns
    -------
    trough_pos, peak_pos, zero_pos : np.ndarray[int]
    """
    empty = np.empty(0, dtype=np.int64)
    n = x_narrow.shape[0]
    if n < 3:
        return empty, empty.copy(), empty.copy()

    s = _forward_fill_sign(x_narrow)
    change = np.flatnonzero(np.diff(s) != 0) + 1           # first sample of each segment
    if change.size == 0:
        return empty, empty.copy(), empty.copy()

    seg_starts = np.concatenate(([0], change))
    seg_ids = np.zeros(n, dtype=np.int64)
    seg_ids[change] = 1
    np.cumsum(seg_ids, out=seg_ids)
    n_seg = seg_starts.size
    seg_polarity = s[seg_starts]                            # +/-1 per segment

    # pairs: negative segment i followed by positive segment i+1
    pair = (seg_polarity[:-1] < 0) & (seg_polarity[1:] > 0)
    neg_idx = np.flatnonzero(pair)
    if neg_idx.size == 0:
        return empty, empty.copy(), empty.copy()

    trough_of_seg = _argext_per_segment(x_narrow, seg_ids, n_seg, 'min')
    peak_of_seg = _argext_per_segment(x_narrow, seg_ids, n_seg, 'max')

    trough_pos = trough_of_seg[neg_idx]
    peak_pos = peak_of_seg[neg_idx + 1]
    zero_pos = seg_starts[neg_idx]                          # pos->neg crossing

    # refine on the (non-band-passed) reference within +/- half period of f_high
    half_win = int(round((1.0 / f_high) * fs / 2))
    trough_pos = _refine_positions(x_ref, trough_pos, half_win, 'min')
    peak_pos = _refine_positions(x_ref, peak_pos, half_win, 'max')

    # duration gate: half a period of f_high < trough->peak < half a period of f_low
    dur = peak_pos - trough_pos
    lo_bound = (1.0 / f_high) * fs / 2
    hi_bound = (1.0 / f_low) * fs / 2
    keep = (dur > lo_bound) & (dur < hi_bound)
    return trough_pos[keep], peak_pos[keep], zero_pos[keep]


def detect_waves(x, fs, fband=(0.5, 4.0), measure_on=None):
    """
    Detect waves in a single 1-D signal and return their positions and morphology.

    Parameters
    ----------
    x : np.ndarray
        1-D signal.
    fs : float
        Sampling frequency (Hz).
    fband : (float, float)
        (low, high) band in Hz that defines the waves to detect.
    measure_on : np.ndarray, optional
        Signal the amplitudes/slopes are read from (same length as ``x``). Detection
        always runs on ``x``; when ``measure_on`` is given, morphology is measured on it
        (e.g. detect on a narrow band, measure on a 0.5-35 Hz broadband trace). Defaults
        to the drift-removed ``x``.

    Returns
    -------
    dict
        Keys: ``min_pos, min_val, max_pos, max_val, zero_pos`` (arrays over waves) and the
        derived per-wave arrays ``pk2pk, delta_t, upslope, down_dur, downslope``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    f_low, f_high = float(fband[0]), float(fband[1])

    x0 = x - np.nanmean(x)
    # drift removal (high-pass at f_low) via the same brick-wall FFT filter
    x_ref = x0 - _bandpass_fft(x0, fs, 0.0, f_low)
    x_narrow = _bandpass_fft(x0, fs, f_low, f_high)

    if measure_on is None:
        amp = x_ref
    else:
        amp = np.asarray(measure_on, dtype=np.float64).ravel()
        if amp.shape[0] != x0.shape[0]:
            raise ValueError(
                f'measure_on length ({amp.shape[0]}) must match x length ({x0.shape[0]}).')
        amp = amp - np.nanmean(amp)

    trough_pos, peak_pos, zero_pos = _find_wave_pairs(x_narrow, x_ref, fs, f_low, f_high)

    min_val = amp[trough_pos]
    max_val = amp[peak_pos]
    pk2pk = max_val - min_val
    delta_t = (peak_pos - trough_pos) / fs
    down_dur = (trough_pos - zero_pos) / fs
    with np.errstate(divide='ignore', invalid='ignore'):
        upslope = np.where(delta_t > 0, pk2pk / delta_t, np.nan)
        downslope = np.where(down_dur > 0, -min_val / down_dur, np.nan)

    return {
        'min_pos': trough_pos, 'min_val': min_val,
        'max_pos': peak_pos, 'max_val': max_val,
        'zero_pos': zero_pos,
        'pk2pk': pk2pk, 'delta_t': delta_t,
        'upslope': upslope, 'down_dur': down_dur, 'downslope': downslope,
    }


# ----------------------------------------------------------------------------------
# detector
# ----------------------------------------------------------------------------------
class WaveDetector:
    """
    Band-limited wave detector and windowed feature extractor.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    fband : (float, float)
        Single ``(low, high)`` band in Hz. One detector detects in one band; run several
        detectors for several bands. Default ``(0.5, 4.0)`` (delta).
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
        If True, prepend a ``DATA_RATE`` feature (fraction of non-NaN samples per window).
    n_processes : int
        Parallelise detection across signals for 2-D / list input. Default ``1``.
    """

    __version__ = '2.0.0'

    _SHAPE_FEATURES = ('WAVE_PK2PK_MEAN', 'WAVE_SLOPE_MEAN', 'WAVE_DELTA_T_MEAN',
                       'WAVE_MIN_MEAN', 'WAVE_MAX_MEAN')

    def __init__(self, fs, fband=(0.5, 4.0), segm_size=None, overlap=0.0,
                 slope='downslope', amplitude_threshold=None,
                 datarate=False, n_processes=1,
                 cutoff_low=None, cutoff_high=None):
        # backward-compatible aliases for the old (cutoff_low, cutoff_high) signature
        if cutoff_low is not None or cutoff_high is not None:
            fband = (cutoff_low if cutoff_low is not None else fband[0],
                     cutoff_high if cutoff_high is not None else fband[1])

        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError(f'fs must be a positive number. Got: {fs}')
        if len(fband) != 2 or not (0 <= fband[0] < fband[1]):
            raise ValueError(f'fband must be (low, high) with 0 <= low < high. Got: {fband}')
        if fband[1] >= fs / 2:
            raise ValueError(f'fband high ({fband[1]}) must be below Nyquist ({fs / 2}).')
        if segm_size is not None and (segm_size <= 0 or not np.isfinite(segm_size)):
            raise ValueError(f'segm_size must be a positive finite number of seconds or None. Got: {segm_size}')
        if segm_size is not None and not (0 <= overlap < segm_size):
            raise ValueError(f'overlap must be in [0, segm_size). Got: {overlap}')
        if slope not in ('downslope', 'upslope'):
            raise ValueError(f"slope must be 'downslope' or 'upslope'. Got: {slope!r}")
        if not isinstance(n_processes, int) or n_processes < 1:
            raise ValueError(f'n_processes must be a positive integer. Got: {n_processes}')

        self.fs = float(fs)
        self.fband = (float(fband[0]), float(fband[1]))
        self.segm_size = segm_size
        self.overlap = overlap
        self.slope = slope
        self.amplitude_threshold = amplitude_threshold
        self.datarate = datarate
        self.n_processes = n_processes

    # -- backward-compatible read-only aliases ------------------------------------
    @property
    def cutoff_low(self):
        return self.fband[0]

    @property
    def cutoff_high(self):
        return self.fband[1]

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
            A single detection dict for 1-D input, otherwise one dict per signal.
            Signals are detected **independently** -- positions index into that signal.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        worker = partial(detect_waves, fs=self.fs, fband=self.fband)

        if self.n_processes > 1 and len(signals) > 1:
            with multiprocessing.Pool(self.n_processes) as pool:
                results = pool.starmap(
                    partial(_detect_one, fs=self.fs, fband=self.fband, thr=self.amplitude_threshold),
                    list(zip(signals, measures)))
        else:
            results = [_detect_one(s, m, fs=self.fs, fband=self.fband, thr=self.amplitude_threshold)
                       for s, m in zip(signals, measures)]

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
            WAVE_MIN_MEAN, WAVE_MAX_MEAN]``. Empty windows give ``WAVE_RATE = 0`` and NaN
            shape features.
        """
        signals, measures, single = self._as_signal_list(x, measure_on)
        detections = self.detect(x, measure_on)
        if single:
            detections = [detections]

        names = (['DATA_RATE'] if self.datarate else []) + ['WAVE_RATE'] + list(self._SHAPE_FEATURES)
        per_signal = [self._features_for_signal(sig, det) for sig, det in zip(signals, detections)]

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

    def _features_for_signal(self, sig, det):
        n = sig.shape[0]
        starts, n_segm = self._window_starts(n)
        win_dur = n_segm / self.fs

        trough = det['min_pos']
        slope = det['downslope'] if self.slope == 'downslope' else det['upslope']
        row = {}
        rate, pk2pk, slp, dt, mn, mx, drate = [], [], [], [], [], [], []
        for s in starts:
            in_win = (trough >= s) & (trough < s + n_segm)
            k = int(in_win.sum())
            rate.append(k / win_dur)
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
                seg = sig[s:s + n_segm]
                drate.append(np.mean(~np.isnan(seg)) if seg.size else np.nan)

        row['WAVE_RATE'] = np.asarray(rate)
        row['WAVE_PK2PK_MEAN'] = np.asarray(pk2pk)
        row['WAVE_SLOPE_MEAN'] = np.asarray(slp)
        row['WAVE_DELTA_T_MEAN'] = np.asarray(dt)
        row['WAVE_MIN_MEAN'] = np.asarray(mn)
        row['WAVE_MAX_MEAN'] = np.asarray(mx)
        if self.datarate:
            row['DATA_RATE'] = np.asarray(drate)
        return row

    def _as_signal_list(self, x, measure_on):
        single = False
        if isinstance(x, np.ndarray) and x.ndim == 1:
            signals = [np.asarray(x, dtype=np.float64)]
            single = True
        elif isinstance(x, np.ndarray):
            if x.ndim != 2:
                raise ValueError(f"Input 'x' must be 1-D or 2-D. Got {x.ndim}-D.")
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


def _detect_one(sig, measure, fs, fband, thr):
    """Detect on one signal and apply the optional negative-peak amplitude threshold."""
    det = detect_waves(sig, fs=fs, fband=fband, measure_on=measure)
    if thr is not None and det['min_pos'].size:
        keep = (-det['min_val']) >= thr
        det = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v)
               for k, v in det.items()}
    return det
