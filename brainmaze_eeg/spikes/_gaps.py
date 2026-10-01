# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gap (NaN/inf) helpers used by :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector`.

Three steps around an unchanged detector:

1. :func:`find_gaps` -- runs of non-finite samples (NaN or +/-inf) of the **original**
   signal, as ``[start, stop)`` sample indices.
2. :func:`fill_gaps` -- make the signal finite so filters/FFT/Hilbert can run:

   - gaps up to ``max_interp_s`` (default 0.1 s here; the wrapper passes its own
     ``short_gap_s``, 0.02 s) are linearly interpolated;
   - longer gaps, ``method='pink'``: 1/f noise scaled to the robust amplitude (MAD) of the
     neighbouring ``context_s`` seconds, offset to the local level (bridged linearly between
     the two sides) and cross-faded with a raised-cosine taper of ``taper_s`` seconds into
     the mirror image of the neighbouring signal at each edge (continuous at the edges);
   - longer gaps, ``method='mirror'``: the neighbouring signal mirrored into the gap from both
     sides, the two images cross-faded over the whole gap (keeps the local spectrum). Real
     events next to the gap are copied into it. Detections inside the gap are removed by
     the wrapper, but the mirrored copies still compete with real maxima in peak selection
     (``find_peaks(distance=...)``) just outside the gap: on spike-dense real data the
     extra/missing detections sit 0.10-0.16 s from the edge, which is why the wrapper's
     ``edge_margin_s`` default is 0.2 s (independent review of PR #67, R7);
   - ``method='linear'``: straight line for every gap (not recommended for gaps > ~0.1 s
     before a background-modelling detector such as Janca).

   Each gap draws its noise from its own generator, seeded from ``seed`` and the gap's
   position and neighbouring data, so fills are reproducible, independent across channels
   and independent of the other gaps.
3. :func:`mask_in_gaps` / :func:`drop_in_gaps` -- remove detections inside a gap or within
   ``margin_s`` of it. Detections on filled samples are never real, whatever the fill.

Relation to ``brainmaze_utils.gaps``
------------------------------------
The canonical gap helpers of the BrainMaze family are being added to brainmaze-utils
(``brainmaze_utils.gaps``, PR bnelair/brainmaze-utils#26, to ship in brainmaze-utils 2.1.0).
This module is a deliberately thin, self-contained stand-in with the same names, signatures
and semantics as that module's final API: ``find_gaps``, ``gap_intervals``,
``fill_gaps(x, fs, *, max_interp_s, method, context_s, taper_s, beta, seed, axis, all_nan,
copy)``, ``pink_noise``, and ``mask_in_gaps``/``drop_in_gaps(det, gaps, fs, *, units,
gap_units, margin_s=0.1, end=None)`` with explicit, separate units for the detections and the
gaps (both required). The one difference: there is **no**
``'spectral'`` fill here (the default there), so this module's default ``method`` is
``'mirror'``. :class:`~brainmaze_eeg.spikes.gap_aware.GapAwareSpikeDetector` always passes
``method`` explicitly, so switching to ``from brainmaze_utils.gaps import ...`` is a one-line
change (follow-up, once 2.1.0 is released; then benchmark ``'spectral'`` as the wrapper
default).
"""

import hashlib
import warnings

import numpy as np

__all__ = ['FILL_METHODS', 'find_gaps', 'gap_intervals', 'fill_gaps', 'pink_noise',
           'mask_in_gaps', 'drop_in_gaps', 'gap_mask']

FILL_METHODS = ('pink', 'mirror', 'linear')


def find_gaps(x):
    """
    Runs of non-finite samples (NaN, +inf, -inf) in a 1-D signal.

    Returns
    -------
    np.ndarray, shape (n_gaps, 2), int64
        ``[start, stop)`` sample indices of each run (``stop`` exclusive), in order.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'find_gaps expects a 1-D signal, got shape {x.shape}')
    d = np.diff(np.concatenate(([0], (~np.isfinite(x)).astype(np.int8), [0])))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


def gap_intervals(x, fs):
    """Gaps of a 1-D signal as ``[start, stop)`` times in **seconds**, shape (n_gaps, 2)."""
    _check_fs(fs)
    return find_gaps(x) / float(fs)


def _check_fs(fs):
    try:
        ok = np.isfinite(fs) and fs > 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f'fs must be a finite number > 0, got {fs!r}')


def gap_mask(gaps, n_samples):
    """Boolean ``(n_samples,)`` mask, True on the samples covered by ``gaps`` (samples)."""
    m = np.zeros(n_samples, dtype=bool)
    for s, e in np.asarray(gaps, dtype=np.int64).reshape(-1, 2):
        m[s:e] = True
    return m


def pink_noise(n, beta=1.0, fmin_bins=1, rng=None):
    """Zero-mean, unit-variance ``1/f**beta`` noise of length ``n`` (spectral synthesis).
    Frequency bins below ``fmin_bins`` are zeroed."""
    rng = np.random.default_rng(rng)
    if n <= 1:
        return np.zeros(max(n, 0))
    k = np.arange(n // 2 + 1, dtype=float)
    amp = np.zeros_like(k)
    keep = k >= max(fmin_bins, 1)
    amp[keep] = k[keep] ** (-beta / 2.0)
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


def _reflect(k, length):
    """Triangle-wave index 0..L-1, L-1..0, 0.. so a mirror image can exceed its context."""
    if length <= 1:
        return np.zeros_like(k)
    k = k % (2 * length)
    return np.where(k < length, k, 2 * length - 1 - k)


def _root_entropy(seed):
    """Entropy of the root SeedSequence of one fill operation (drawn once per call)."""
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(0, 2 ** 63))       # one draw: per-gap streams stay independent
    if isinstance(seed, np.random.SeedSequence):
        return seed.entropy
    if seed is None:
        return np.random.SeedSequence().entropy
    if isinstance(seed, np.random.RandomState):
        raise TypeError('seed must be None, an int, a sequence of ints, a SeedSequence or a '
                        'Generator (not RandomState)')
    return seed


def _gap_rng(entropy, s, e, context):
    """Generator for one gap: root entropy + (start, stop, hash of the neighbouring data)."""
    h = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(context).tobytes(),
                                       digest_size=8).digest(), 'little')
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(int(s), int(e), h)))


def _fill_1d(x, fs, max_interp_s, method, context_s, taper_s, beta, entropy):
    y = np.array(x, dtype=np.float64, copy=True)
    gaps = find_gaps(y)
    n_total = y.size
    if gaps.size == 0 or (len(gaps) == 1 and gaps[0, 0] == 0 and gaps[0, 1] == n_total):
        return y
    max_interp = int(np.floor(max_interp_s * fs + 1e-9))
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
        # context: finite samples next to the gap (left side already filled -> finite)
        left = y[max(s - ctx, 0):s]
        right = y[e:min(e + ctx, nxt)]
        lvl_a = np.median(left[-near:]) if left.size else np.median(right[:near])
        lvl_b = np.median(right[:near]) if right.size else np.median(left[-near:])
        base = lvl_a + (lvl_b - lvl_a) * r
        if method == 'mirror':
            k = np.arange(n)
            imgs = []
            if left.size:
                imgs.append(y[s - 1 - _reflect(k, left.size)] - lvl_a)
            if right.size:
                imgs.append(y[e + _reflect(n - 1 - k, right.size)] - lvl_b)
            if len(imgs) == 2:
                w = _raised_cosine(n)
                dev = (w * imgs[0] + (1 - w) * imgs[1]) / np.sqrt(w ** 2 + (1 - w) ** 2)
            else:
                dev = imgs[0]
            y[s:e] = base + dev
            continue
        # pink: independent, reproducible stream per gap
        context = np.concatenate([left, right])
        rng = _gap_rng(entropy, s, e, context)
        fill = base + _robust_sd(context) * pink_noise(n, beta, rng=rng)
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


def fill_gaps(x, fs, *, max_interp_s=0.1, method='mirror', context_s=30.0, taper_s=0.5,
              beta=1.0, seed=0, axis=-1, all_nan='keep', copy=True):
    """
    Fill the non-finite gaps of a signal (see module docstring).

    Parameters
    ----------
    x : array_like
        Signal, 1-D ``(n_samples,)`` or N-D with time along ``axis``; every other index is
        filled independently (a row of an N-D array is filled exactly as the same 1-D
        signal).
    fs : float
        Sampling frequency (Hz).
    max_interp_s : float
        Gaps up to this length (s) are linearly interpolated (default 0.1 s).
    method : {'mirror', 'pink', 'linear'}
        Fill of longer gaps (default ``'mirror'``; ``brainmaze_utils.gaps`` additionally has
        ``'spectral'``, its default).
    context_s : float
        Seconds of valid data on each side used for amplitude, level and mirroring
        (default 30, as in ``brainmaze_utils.gaps``; the wrapper passes 10 explicitly, its
        tuned value).
    taper_s : float
        ``'pink'``: raised-cosine cross-fade length (s) into the mirrored signal at each
        edge, capped at half the gap (default 0.5).
    beta : float
        ``'pink'``: spectral exponent, ``P(f) ~ 1/f**beta`` (default 1).
    seed : None, int, sequence of int, np.random.SeedSequence or np.random.Generator
        Noise seed (default 0: reproducible). Each gap gets its own stream derived from the
        seed, the gap's position and a hash of its neighbouring data, so channels with the
        same seed still get independent noise.
    axis : int
        Time axis (default -1).
    all_nan : {'keep', 'zero', 'raise'}
        A channel without any finite sample: left unchanged (``'keep'``), set to zeros
        (``'zero'``) -- both with one warning listing the channels -- or ``ValueError``
        (``'raise'``).
    copy : bool
        ``True`` (default): ``x`` is not modified. ``False``: a writable floating-point
        ndarray is filled in place and returned.

    Returns
    -------
    np.ndarray
        Same shape as ``x``; floating input keeps its dtype, anything else becomes float64.
    """
    _check_fs(fs)
    if method not in FILL_METHODS:
        raise ValueError(f'fill method must be one of {FILL_METHODS}, got {method!r}')
    if all_nan not in ('keep', 'zero', 'raise'):
        raise ValueError(f"all_nan must be 'keep', 'zero' or 'raise', got {all_nan!r}")
    if not (np.isfinite(max_interp_s) and max_interp_s >= 0 and np.isfinite(taper_s)
            and taper_s >= 0 and context_s > 0 and np.isfinite(beta)):
        raise ValueError('max_interp_s and taper_s must be finite >= 0, context_s > 0 and '
                         'beta finite')
    x_in = x
    x = np.asarray(x)
    if np.iscomplexobj(x) or not (np.issubdtype(x.dtype, np.number)
                                  or np.issubdtype(x.dtype, np.bool_)):
        raise TypeError(f'x must be a real numeric array, got dtype {x.dtype}')
    if x.ndim == 0:
        raise ValueError('x must have at least one dimension')
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f'axis {axis} out of range for {x.ndim}-D input')
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    in_place = (not copy and isinstance(x_in, np.ndarray) and x_in.dtype == dtype
                and x_in.flags.writeable)
    out = x_in if in_place else np.array(x, dtype=dtype, copy=True)
    entropy = _root_entropy(seed)
    view = np.moveaxis(out, axis, -1)
    dead = []
    for idx in np.ndindex(view.shape[:-1]):
        row = view[idx]
        if row.size == 0 or np.isfinite(row).all():
            continue
        if not np.isfinite(row).any():
            dead.append(idx)
            if all_nan == 'zero':
                row[...] = 0
            continue
        row[...] = _fill_1d(row, fs, max_interp_s, method, context_s, taper_s, beta, entropy)
    if dead:
        if all_nan == 'raise':
            raise ValueError(f'channel(s) {dead} contain no finite sample')
        warnings.warn(f'fill_gaps: channel(s) {dead} contain no finite sample; '
                      + ('left unchanged' if all_nan == 'keep' else 'set to zeros'),
                      RuntimeWarning, stacklevel=2)
    return out


_INTEGRAL_TOL = 1e-6             # |v - round(v)| accepted as an integer sample index


def _to_seconds(name, v, units, fs, units_name):
    """Positions in ``units`` -> float seconds (same validation as brainmaze_utils.gaps)."""
    v = np.asarray(v)
    if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError(f'{name} must be real numbers, got dtype {v.dtype}')
    vf = v.astype(np.float64)
    if not np.isfinite(vf).all():
        raise ValueError(f'{name} contains NaN/inf')
    if units == 'seconds':
        if vf.size and np.issubdtype(v.dtype, np.integer):
            raise ValueError(
                f"{name} is integer-typed (sample indices?) but {units_name}='seconds'. "
                f"Use {units_name}='samples' for sample indices; if they really are whole "
                f"seconds, pass them as floats (np.asarray({name}, float)).")
        return vf
    if not np.issubdtype(v.dtype, np.integer):
        r = np.round(vf)
        if np.any(np.abs(vf - r) > _INTEGRAL_TOL):
            raise ValueError(
                f"{name} has non-integer values but {units_name}='samples'. Are they in "
                f"seconds? Then use {units_name}='seconds'. (Values within {_INTEGRAL_TOL:g} "
                "of an integer, e.g. t * fs, are accepted as sample indices.)")
        vf = r
    return vf / fs


def mask_in_gaps(det, gaps, fs, *, units, gap_units, margin_s=0.1, end=None):
    """
    Boolean mask of detections overlapping a gap widened by ``margin_s`` on each side.

    Parameters
    ----------
    det : array_like
        Detection positions, or interval starts, in ``units``.
    gaps : array_like, shape (n_gaps, 2)
        ``[start, stop)`` of each gap in ``gap_units``: :func:`find_gaps` gives samples,
        :func:`gap_intervals` gives seconds.
    fs : float
        Sampling frequency (Hz), required.
    units : {'seconds', 'samples'}
        Units of ``det`` (and ``end``); required, never inferred.
    gap_units : {'seconds', 'samples'}
        Units of ``gaps``; required, never inferred. Integer-typed values with
        ``'seconds'`` raise ``ValueError`` (pass whole seconds as floats); with
        ``'samples'``, float values must be integral within 1e-6 (``t * fs`` is fine) or
        ``ValueError`` is raised (a units mix-up must not silently mask nothing).
    margin_s : float
        Exclusion margin in **seconds** (default 0.1).
    end : array_like, optional
        Interval ends (same shape as ``det``, ``end >= det``).

    Returns
    -------
    np.ndarray of bool
        True where ``[det, end]`` overlaps ``[gap_start - margin, gap_stop + margin)``.
    """
    for nm, u in (('units', units), ('gap_units', gap_units)):
        if u not in ('samples', 'seconds'):
            raise ValueError(f"{nm} must be 'seconds' or 'samples', got {u!r}")
    _check_fs(fs)
    fs = float(fs)
    if not (np.isfinite(margin_s) and margin_s >= 0):
        raise ValueError(f'margin_s must be finite >= 0, got {margin_s}')
    g_raw = np.asarray(gaps)
    if g_raw.size == 0:
        g_raw = g_raw.reshape(0, 2)
    if g_raw.ndim != 2 or g_raw.shape[1] != 2:
        raise ValueError(f'gaps must have shape (n_gaps, 2), got {g_raw.shape}')
    g = _to_seconds('gaps', g_raw, gap_units, fs, 'gap_units')
    if np.any(g[:, 1] < g[:, 0]):
        raise ValueError('gaps must have stop >= start')
    t0 = np.atleast_1d(_to_seconds('det', det, units, fs, 'units'))
    if end is None:
        t1 = t0
    else:
        t1 = np.atleast_1d(_to_seconds('end', end, units, fs, 'units'))
        if t1.shape != t0.shape:
            raise ValueError(f'end has shape {t1.shape}, det has shape {t0.shape}')
        if np.any(t1 < t0):
            raise ValueError('end must be >= det for every detection')
    out = np.zeros(t0.shape, dtype=bool)
    if g.size == 0 or t0.size == 0:
        return out
    lo = g[:, 0] - margin_s
    order = np.argsort(lo)
    lo, hi = lo[order], np.maximum.accumulate(g[order, 1] + margin_s)
    k = np.searchsorted(lo, t1, side='right') - 1          # last widened gap starting <= end
    ok = k >= 0
    out[ok] = t0[ok] < hi[k[ok]]
    return out


def drop_in_gaps(det, gaps, fs, *, units, gap_units, margin_s=0.1, end=None):
    """
    ``det`` without the detections flagged by :func:`mask_in_gaps` (original dtype), or
    ``(det, end)`` when ``end`` is given.
    """
    det_a = np.atleast_1d(np.asarray(det))
    keep = ~mask_in_gaps(det_a, gaps, fs, units=units, gap_units=gap_units,
                         margin_s=margin_s, end=end)
    if end is None:
        return det_a[keep]
    return det_a[keep], np.atleast_1d(np.asarray(end))[keep]
