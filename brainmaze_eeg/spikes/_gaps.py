# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NaN / gap handling shared by the spike detectors (pre- and post-processing).

The detectors never see a NaN. Missing data are handled in two dedicated steps around the
unchanged detection algorithm:

1. **Pre** -- :func:`prepare_signal`: find the gaps (runs of NaN) per channel on the
   **original** signal (:func:`find_gaps`) and, with ``nan_policy='fill'``, fill them with
   :func:`fill_gaps`:

   - gaps up to ``max_interp_s`` (default 0.1 s) are linearly interpolated between the
     edge values;
   - longer gaps (``method='pink'``, default) get 1/f noise scaled to the robust amplitude
     (MAD) of the neighbouring ``context_s`` seconds, offset to the local level (bridged
     linearly between the two sides), and cross-faded with a raised-cosine taper of
     ``taper_s`` seconds into the *mirror image* of the neighbouring signal at each edge, so
     the filled signal is continuous at the gap edges (no step that would ring through the
     filters, no silent stretch that would pull the detectors' running background down);
   - ``method='linear'``: straight line for every gap (not recommended for gaps > ~0.1 s
     before a background-modelling detector such as Janca).

   With ``nan_policy='raise'`` any NaN raises :class:`ValueError`.
2. **Post** -- :func:`in_gap_mask` / :func:`mask_in_gaps`: flag detections that fall inside
   a gap of the original signal or within ``margin_s`` of one, so the detector drops them.
   Detections on filled samples are never real, whatever the fill method.

Without this, a single NaN propagates through ``filtfilt``/FFT/Hilbert to the whole channel
(Janca: zero detections, silently) or through ``np.median`` to the block-scaling factor of
every channel (Barkmeier). ``+/-inf`` is never treated as a gap: it always raises.

Relation to ``brainmaze_utils.gaps``
------------------------------------
The canonical gap helpers of the BrainMaze family are being added to brainmaze-utils
(``brainmaze_utils.gaps``, PR bnelair/brainmaze-utils#26, not yet released). This module is a
deliberately thin, self-contained stand-in with **the same function names and semantics**
(``find_gaps`` -> ``(n_gaps, 2)`` ``[start, stop)`` samples; ``fill_gaps(x, fs,
max_interp_s=0.1, method='pink', context_s=10, taper_s=0.5, beta=1, seed=0)``;
``mask_in_gaps``/``drop_in_gaps(times_s, gaps_samples, fs, margin_s=0.1)`` with explicit
units) so that the switch is a
one-line import change once brainmaze-utils with ``gaps`` is released. Only the ``'pink'``
and ``'linear'`` fills are provided here (the ``'mirror'`` fill will come with the switch).
Fills are seeded (``seed=0``) and therefore reproducible.

The defaults come from a benchmark on 1 h of scalp EEG (500 Hz) with 23 gaps per length and
IED-like transients injected 0.15-1.2 s from the gap edges: pink fill + dropping detections
within 0.1 s of a gap gave 0-2 false detections around 23 gaps of 0.5-60 s (linear fill:
113-312 for gaps >= 2 s) and kept 100 % of the transients 0.15 s from the edge.
"""

import warnings

import numpy as np

__all__ = ['NAN_POLICIES', 'find_gaps', 'fill_gaps', 'pink_noise', 'mask_in_gaps',
           'drop_in_gaps', 'prepare_signal', 'in_gap_mask', 'gap_sample_mask']

NAN_POLICIES = ('fill', 'raise')
FILL_METHODS = ('pink', 'linear')


# ------------------------------------------------------------------ brainmaze_utils.gaps API
def find_gaps(x):
    """
    Runs of NaN samples in a 1-D signal.

    Returns
    -------
    np.ndarray, shape (n_gaps, 2), int64
        ``[start, stop)`` sample indices of each NaN run (``stop`` exclusive), in order.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'find_gaps expects a 1-D signal, got shape {x.shape}')
    d = np.diff(np.concatenate(([0], np.isnan(x).astype(np.int8), [0])))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


def pink_noise(n, beta=1.0, rng=None):
    """Zero-mean, unit-variance ``1/f**beta`` noise of length ``n`` (spectral synthesis)."""
    rng = np.random.default_rng() if rng is None else rng
    if n <= 1:
        return np.zeros(max(n, 0))
    k = np.arange(n // 2 + 1, dtype=float)
    amp = np.zeros_like(k)
    amp[1:] = k[1:] ** (-beta / 2.0)
    y = np.fft.irfft(amp * np.exp(2j * np.pi * rng.random(k.size)), n)
    sd = y.std()
    return (y - y.mean()) / sd if sd > 0 else np.zeros(n)


def _robust_sd(v):
    if v.size < 2:
        return 0.0
    sd = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(sd if sd > 0 else v.std())


def _raised_cosine(n):
    """Weights going 1 -> 0 over ``n`` samples (end points excluded)."""
    return 0.5 * (1 + np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))


def _fill_1d(x, fs, max_interp_s, method, context_s, taper_s, beta, seed, stream):
    y = np.array(x, dtype=np.float64, copy=True)
    gaps = find_gaps(y)
    n_total = y.size
    if gaps.size == 0 or (len(gaps) == 1 and gaps[0, 0] == 0 and gaps[0, 1] == n_total):
        return y
    max_interp = int(round(max_interp_s * fs))
    ctx = max(int(round(context_s * fs)), 2)
    taper = int(round(taper_s * fs))
    near = max(min(int(round(0.5 * fs)), ctx), 1)
    nxt_start = np.r_[gaps[1:, 0], n_total]
    for (s, e), nxt in zip(gaps, nxt_start):
        n = e - s
        has_l, has_r = s > 0, e < n_total
        r = np.arange(1, n + 1) / (n + 1)
        if n <= max_interp or method == 'linear':
            a = y[s - 1] if has_l else y[e]
            b = y[e] if has_r else y[s - 1]
            y[s:e] = a + (b - a) * r
            continue
        # context: valid samples next to the gap (left side is already filled -> finite)
        left = y[max(s - ctx, 0):s]
        right = y[e:min(e + ctx, nxt)]
        sd = _robust_sd(np.concatenate([left, right]))
        lvl_a = np.median(left[-near:]) if left.size else np.median(right[:near])
        lvl_b = np.median(right[:near]) if right.size else np.median(left[-near:])
        # independent, reproducible noise stream per (seed, channel/stream, gap start)
        rng = np.random.default_rng(None if seed is None else [int(seed), int(stream), int(s)])
        fill = lvl_a + (lvl_b - lvl_a) * r + sd * pink_noise(n, beta, rng)
        # raised-cosine cross-fade into the mirrored neighbouring signal at each edge
        t = min(taper, n // 2) if (has_l and has_r) else min(taper, n)
        tl = min(t, left.size)
        if has_l and tl > 0:
            w = _raised_cosine(tl)
            fill[:tl] = w * y[s - 1 - np.arange(tl)] + (1 - w) * fill[:tl]
        tr = min(t, right.size)
        if has_r and tr > 0:
            w = _raised_cosine(tr)[::-1]
            fill[n - tr:] = w * y[e + np.arange(tr)][::-1] + (1 - w) * fill[n - tr:]
        y[s:e] = fill
    return y


def fill_gaps(x, fs, max_interp_s=0.1, method='pink', context_s=10.0, taper_s=0.5,
              beta=1.0, seed=0, stream=0):
    """
    Fill NaN gaps of a 1-D signal so a detector can run on it (see module docstring).

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Signal; not modified.
    fs : float
        Sampling frequency (Hz).
    max_interp_s : float
        Gaps up to this length (s) are linearly interpolated (default 0.1 s).
    method : {'pink', 'linear'}
        Fill of longer gaps (default ``'pink'``).
    context_s : float
        Seconds of valid data on each side used for amplitude and level (default 10).
    taper_s : float
        Raised-cosine cross-fade length (s) into the mirrored signal at each edge, capped at
        half the gap (default 0.5).
    beta : float
        Spectral exponent of the noise, ``P(f) ~ 1/f**beta`` (default 1, pink).
    seed : int or None
        Noise seed (default 0: reproducible). ``None`` = random.
    stream : int
        Stream index (e.g. the channel number). Each gap is filled from its own generator
        seeded with ``(seed, stream, gap_start)``, so channels get independent noise even
        where their gaps coincide, and a gap's fill does not depend on other gaps.

    Returns
    -------
    np.ndarray (float64)
        Filled copy. An all-NaN signal is returned unchanged.
    """
    if not fs > 0:
        raise ValueError(f'fs must be > 0, got {fs}')
    if method not in FILL_METHODS:
        raise ValueError(f'fill method must be one of {FILL_METHODS}, got {method!r} '
                         "('mirror' becomes available with brainmaze_utils.gaps)")
    if not (max_interp_s >= 0 and taper_s >= 0 and context_s > 0):
        raise ValueError('max_interp_s and taper_s must be >= 0 and context_s > 0')
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f'fill_gaps expects a 1-D signal, got shape {x.shape}')
    return _fill_1d(x, fs, max_interp_s, method, context_s, taper_s, beta, seed, stream)


def mask_in_gaps(times_s, gaps_samples, fs, margin_s=0.1):
    """
    Boolean mask of detections that fall in a gap widened by ``margin_s`` on each side.

    Units are explicit (no inference): ``times_s`` in **seconds**, ``gaps_samples`` as
    returned by :func:`find_gaps` (``[start, stop)`` **sample indices**), ``fs`` in Hz
    (required). A detection at time ``t`` is masked iff
    ``start/fs - margin_s <= t < stop/fs + margin_s``.
    """
    if fs is None or not fs > 0:
        raise ValueError(f'fs must be > 0 Hz (gaps are sample indices), got {fs!r}')
    if not margin_s >= 0:
        raise ValueError(f'margin_s must be >= 0, got {margin_s}')
    t = np.atleast_1d(np.asarray(times_s, dtype=float))
    g = np.asarray(gaps_samples, dtype=float).reshape(-1, 2) / float(fs)
    if g.size == 0 or t.size == 0:
        return np.zeros(t.shape, dtype=bool)
    lo = g[:, 0] - margin_s
    order = np.argsort(lo)
    lo, hi = lo[order], np.maximum.accumulate(g[order, 1] + margin_s)
    k = np.searchsorted(lo, t, side='right') - 1
    out = np.zeros(t.shape, dtype=bool)
    ok = k >= 0
    out[ok] = t[ok] < hi[k[ok]]
    return out


def drop_in_gaps(times_s, gaps_samples, fs, margin_s=0.1):
    """``times_s`` (seconds) without the detections flagged by :func:`mask_in_gaps`."""
    t = np.atleast_1d(np.asarray(times_s, dtype=float))
    return t[~mask_in_gaps(t, gaps_samples, fs, margin_s)]


# ---------------------------------------------------------------- detector-facing helpers
def prepare_signal(x, fs, nan_policy='fill', fill_kwargs=None):
    """
    Pre-processing step: validate finiteness and fill NaN gaps of every channel.

    Parameters
    ----------
    x : np.ndarray, shape (n_channels, n_samples)
        Signal, time along the last axis.
    fs : float
        Sampling frequency (Hz).
    nan_policy : {'fill', 'raise'}
        See module docstring.
    fill_kwargs : dict, optional
        Keyword arguments for :func:`fill_gaps` (``max_interp_s``, ``method``,
        ``context_s``, ``taper_s``, ``beta``, ``seed``); ``stream`` is set to the channel
        index so channels are filled with independent noise.

    Returns
    -------
    y : np.ndarray, shape (n_channels, n_samples)
        Finite signal (a filled copy when there were gaps). All-NaN channels are zeros.
    gaps : list of np.ndarray
        Per channel, ``(n_gaps, 2)`` ``[start, stop)`` sample indices of the NaN runs of the
        **original** signal.
    dead : np.ndarray of bool, shape (n_channels,)
        True for channels without a single finite sample (they produce no detections; a
        ``RuntimeWarning`` is issued).
    """
    if nan_policy not in NAN_POLICIES:
        raise ValueError(f'nan_policy must be one of {NAN_POLICIES}, got {nan_policy!r}')
    if np.isinf(x).any():
        raise ValueError('signal contains +/-inf; only NaN is accepted as a gap marker')
    nan = np.isnan(x)
    n_ch = x.shape[0]
    gaps = [np.zeros((0, 2), dtype=np.int64) for _ in range(n_ch)]
    dead = np.zeros(n_ch, dtype=bool)
    if not nan.any():
        return x, gaps, dead
    if nan_policy == 'raise':
        bad = np.flatnonzero(nan.any(axis=1)).tolist()
        raise ValueError(f"signal contains NaN (channels {bad}); use nan_policy='fill' or "
                         'remove the gaps first')
    kw = dict(fill_kwargs or {})
    kw.pop("stream", None)                  # set per channel below
    dead = nan.all(axis=1)
    y = np.array(x, dtype=np.float64, copy=True)
    for c in np.flatnonzero(nan.any(axis=1)):
        gaps[c] = find_gaps(x[c])
        y[c] = 0.0 if dead[c] else fill_gaps(x[c], fs, stream=c, **kw)
    if dead.any():
        warnings.warn(f'channel(s) {np.flatnonzero(dead).tolist()} are entirely NaN; '
                      'no detections are reported for them', RuntimeWarning, stacklevel=3)
    return y, gaps, dead


def in_gap_mask(times_s, gaps, fs, margin_s):
    """Post-processing step: :func:`mask_in_gaps` for one channel's detection times (s)."""
    times_s = np.atleast_1d(np.asarray(times_s, dtype=float))
    if gaps is None or len(gaps) == 0 or times_s.size == 0:
        return np.zeros(times_s.shape, dtype=bool)
    return mask_in_gaps(times_s, gaps, fs, margin_s)


def gap_sample_mask(gaps, n_samples):
    """Boolean ``(n_samples,)`` mask, True on samples that were NaN in the original signal."""
    m = np.zeros(n_samples, dtype=bool)
    for s, e in np.asarray(gaps, dtype=np.int64).reshape(-1, 2):
        m[s:e] = True
    return m
